from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt, QPoint, QTimer, QObject, QEvent, QMimeData
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)

from bibtex_importer import import_bibtex_file
from bibliography import format_aps_reference, format_aps_references, format_aps_reference_html, format_aps_references_html
from database import Database, doi_to_url, normalize_doi
from journal_abbr import infer_journal_abbr
from metadata_fetcher import fetch_bibtex_record_by_doi, merge_pdf_record_with_bibtex
from pdf_utils import import_pdf_file


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
LIBRARY_DIR = APP_DIR / "library"
PDF_DIR = LIBRARY_DIR / "PDFs"
DB_PATH = DATA_DIR / "papers.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
RESOURCES_DIR = APP_DIR / "resources"
PDF_ICON_PATH = RESOURCES_DIR / "pdf_icon.png"
APP_ICON_PATH = RESOURCES_DIR / "app_icon.ico"
APP_ICON_PNG_PATH = RESOURCES_DIR / "app_icon.png"
BIBTEX_SUFFIXES = {".bib", ".bibtex"}


ROLE_KIND = Qt.UserRole
ROLE_FOLDER_ID = Qt.UserRole + 1


class AutoSavePlainTextEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctrl_enter_callback = None

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ControlModifier:
            if self.ctrl_enter_callback:
                self.ctrl_enter_callback()
                return
        super().keyPressEvent(event)


class UrlLineEdit(QLineEdit):
    """
    URL editing field that can also open the URL.
    - Ctrl + click opens the URL.
    - Double-click opens the URL.
    Normal click still works for text editing.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.IBeamCursor)
        self.setToolTip("Ctrl+click or double-click to open URL in browser")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            url = self.text().strip()
            if url:
                open_url(url)
                return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            url = self.text().strip()
            if url:
                open_url(url)
                return
        super().mouseDoubleClickEvent(event)


class PaperTableWidget(QTableWidget):
    """
    Table widget with Delete-key support.
    Delete is handled only when the paper list itself has focus,
    so editing fields on the right side are not affected.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.delete_key_callback = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            if self.delete_key_callback:
                self.delete_key_callback()
                return
        super().keyPressEvent(event)


class FocusSaveFilter(QObject):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusOut:
            self.callback()
        return False


class PaperManagerWindow(QMainWindow):
    COLUMNS = [
        ("id", "ID"),
        ("title", "Title"),
        ("creators", "Creator"),
        ("year", "Year"),
        ("date", "Date"),
        ("publication", "Publication"),
        ("journal_abbr", "Journal Abbr"),
        ("volume", "Volume"),
        ("pages", "Pages"),
        ("doi", "DOI"),
        ("url", "URL"),
        ("date_added", "Date Added"),
        ("pdf_path", "PDF"),
    ]

    EDIT_FIELDS = [
        ("title", "Title", "line"),
        ("creators", "Creators", "plain"),
        ("year", "Year", "line"),
        ("date", "Date", "line"),
        ("publication", "Publication", "line"),
        ("journal_abbr", "Journal Abbr", "line"),
        ("volume", "Volume", "line"),
        ("pages", "Pages", "line"),
        ("issue", "Issue / Number", "line"),
        ("doi", "DOI", "line"),
        ("url", "URL", "line"),
        ("bibtex_key", "BibTeX Key", "line"),
        ("pdf_path", "PDF Path", "line"),
        ("notes", "Notes", "plain"),
    ]

    def __init__(self) -> None:
        super().__init__()

        self.db = Database(DB_PATH)
        self.pdf_icon = QIcon(str(PDF_ICON_PATH)) if PDF_ICON_PATH.exists() else QIcon()
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        elif APP_ICON_PNG_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PNG_PATH)))
        elif PDF_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(PDF_ICON_PATH)))

        self.current_kind = "all"
        self.current_folder_id: int | None = None
        self.current_paper_id: int | None = None
        self.edit_widgets: dict[str, QLineEdit | QPlainTextEdit] = {}
        self.column_actions: dict[str, QAction] = {}
        self.loading_edit_panel = False
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self.autosave_current_paper)
        self.focus_filter = FocusSaveFilter(self.autosave_current_paper)

        self.settings = self.load_settings()

        self.setWindowTitle("LocalPaperManager")
        self.resize(1600, 850)
        self.setAcceptDrops(True)

        self.create_toolbar()
        self.create_menu()
        self.create_main_layout()
        self.search_shortcut = QShortcut(QKeySequence.Find, self)
        self.search_shortcut.activated.connect(self.focus_search_box)

        self.refresh_folders()
        self.refresh_table()
        self.apply_column_visibility()

    # ---------- settings ----------

    def load_settings(self) -> dict:
        if SETTINGS_PATH.exists():
            try:
                return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_settings(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(self.settings, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- UI construction ----------

    def create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        new_folder_action = QAction("New Folder", self)
        new_folder_action.triggered.connect(self.create_folder_dialog)
        toolbar.addAction(new_folder_action)

        import_pdf_action = QAction("Import PDF", self)
        import_pdf_action.triggered.connect(self.import_pdf_dialog)
        toolbar.addAction(import_pdf_action)

        import_bibtex_action = QAction("Import BibTeX", self)
        import_bibtex_action.triggered.connect(self.import_bibtex_dialog)
        toolbar.addAction(import_bibtex_action)

        fetch_metadata_action = QAction("Fetch BibTeX by DOI", self)
        fetch_metadata_action.triggered.connect(self.fetch_metadata_for_selected_papers)
        toolbar.addAction(fetch_metadata_action)

        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.full_refresh)
        toolbar.addAction(refresh_action)

    def create_menu(self) -> None:
        menu_bar = self.menuBar()
        columns_menu = menu_bar.addMenu("Columns")

        default_visible = self.settings.get("visible_columns")
        if not isinstance(default_visible, list):
            default_visible = [key for key, _ in self.COLUMNS]

        for key, label in self.COLUMNS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(key in default_visible)
            action.toggled.connect(self.on_column_visibility_changed)
            columns_menu.addAction(action)
            self.column_actions[key] = action

        columns_menu.addSeparator()
        show_all = columns_menu.addAction("Show All Columns")
        show_all.triggered.connect(self.show_all_columns)

        hide_rare = columns_menu.addAction("Show Basic Columns")
        hide_rare.triggered.connect(self.show_basic_columns)

    def create_main_layout(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        left = self.create_folder_panel()
        center = self.create_table_panel()
        right = self.create_edit_panel()

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([260, 900, 440])

        self.setCentralWidget(splitter)

    def create_folder_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        label = QLabel("Folders")
        self.folder_list = QListWidget()
        self.folder_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.folder_list.customContextMenuRequested.connect(self.show_folder_context_menu)
        self.folder_list.currentItemChanged.connect(self.on_folder_changed)

        layout.addWidget(label)
        layout.addWidget(self.folder_list)
        return widget

    def create_table_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.status_label = QLabel("PDF or BibTeX files can be dropped into this window. Ctrl+F searches Title and Creator.")

        search_layout = QHBoxLayout()
        search_label = QLabel("Search")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search title or author (Ctrl+F)")
        self.search_box.textChanged.connect(self.on_search_text_changed)
        clear_search_button = QPushButton("Clear")
        clear_search_button.clicked.connect(self.clear_search_box)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        search_layout.addWidget(clear_search_button)

        self.table = PaperTableWidget()
        self.table.delete_key_callback = self.delete_selected_papers
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in self.COLUMNS])
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_paper_context_menu)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.show_header_context_menu)

        widths = {
            "id": 55,
            "title": 330,
            "creators": 220,
            "publication": 190,
            "journal_abbr": 130,
            "doi": 170,
            "url": 240,
            "pdf_path": 70,
        }
        for idx, (key, _) in enumerate(self.COLUMNS):
            if key in widths:
                self.table.setColumnWidth(idx, widths[key])

        layout.addWidget(self.status_label)
        layout.addLayout(search_layout)
        layout.addWidget(self.table)
        return widget

    def create_edit_panel(self) -> QWidget:
        box = QGroupBox("Edit selected paper")
        layout = QVBoxLayout(box)

        help_label = QLabel("Enter: save current field. Ctrl+Enter: save multiline fields.")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        form = QFormLayout()

        for key, label, field_type in self.EDIT_FIELDS:
            if field_type == "plain":
                w = AutoSavePlainTextEdit()
                w.setMaximumHeight(80)
                w.ctrl_enter_callback = self.autosave_current_paper
                w.textChanged.connect(self.schedule_autosave)
            else:
                if key == "url":
                    w = UrlLineEdit()
                    font = w.font()
                    font.setUnderline(True)
                    w.setFont(font)
                else:
                    w = QLineEdit()
                w.returnPressed.connect(self.autosave_current_paper)
                w.editingFinished.connect(self.autosave_current_paper)
                w.textEdited.connect(self.on_edit_text_edited)

            w.installEventFilter(self.focus_filter)
            self.edit_widgets[key] = w
            form.addRow(label, w)

        layout.addLayout(form)

        self.copy_ref_button = QPushButton("Copy APS Reference")
        self.copy_ref_button.clicked.connect(self.copy_selected_paper_reference)
        layout.addWidget(self.copy_ref_button)

        self.open_url_button = QPushButton("Open URL")
        self.open_url_button.clicked.connect(self.open_current_url)
        layout.addWidget(self.open_url_button)

        self.clear_edit_panel()
        return box

    # ---------- columns ----------

    def on_column_visibility_changed(self) -> None:
        self.apply_column_visibility()
        self.settings["visible_columns"] = [
            key for key, action in self.column_actions.items() if action.isChecked()
        ]
        self.save_settings()

    def apply_column_visibility(self) -> None:
        # ID列を全非表示にすると内部処理は問題ないが、ユーザーが戻せるようメニューには残す
        for idx, (key, _) in enumerate(self.COLUMNS):
            action = self.column_actions.get(key)
            visible = True if action is None else action.isChecked()
            self.table.setColumnHidden(idx, not visible)

    def show_all_columns(self) -> None:
        for action in self.column_actions.values():
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
        self.on_column_visibility_changed()

    def show_basic_columns(self) -> None:
        basic = {"title", "creators", "year", "publication", "journal_abbr", "doi", "url", "pdf_path"}
        for key, action in self.column_actions.items():
            action.blockSignals(True)
            action.setChecked(key in basic)
            action.blockSignals(False)
        self.on_column_visibility_changed()

    def show_header_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        for key, label in self.COLUMNS:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.column_actions[key].isChecked())
            action.toggled.connect(self.column_actions[key].setChecked)
        menu.addSeparator()
        show_all = menu.addAction("Show All Columns")
        show_all.triggered.connect(self.show_all_columns)
        menu.exec(self.table.horizontalHeader().mapToGlobal(pos))

    # ---------- Refresh ----------

    def full_refresh(self) -> None:
        self.db.backfill_urls_from_doi()
        self.refresh_folders()
        self.refresh_table()

    def refresh_folders(self) -> None:
        current_kind = self.current_kind
        current_folder_id = self.current_folder_id

        self.folder_list.blockSignals(True)
        self.folder_list.clear()

        all_count = len(self.db.list_papers())
        unfiled_count = len(self.db.list_papers(unfiled=True))

        all_item = QListWidgetItem(f"All Papers ({all_count})")
        all_item.setData(ROLE_KIND, "all")
        all_item.setData(ROLE_FOLDER_ID, None)
        self.folder_list.addItem(all_item)

        unfiled_item = QListWidgetItem(f"Unfiled ({unfiled_count})")
        unfiled_item.setData(ROLE_KIND, "unfiled")
        unfiled_item.setData(ROLE_FOLDER_ID, None)
        self.folder_list.addItem(unfiled_item)

        for folder in self.db.list_folders():
            item = QListWidgetItem(f"{folder['name']} ({folder['paper_count']})")
            item.setData(ROLE_KIND, "folder")
            item.setData(ROLE_FOLDER_ID, int(folder["id"]))
            self.folder_list.addItem(item)

        selected_row = 0
        for i in range(self.folder_list.count()):
            item = self.folder_list.item(i)
            if item.data(ROLE_KIND) == current_kind and item.data(ROLE_FOLDER_ID) == current_folder_id:
                selected_row = i
                break

        self.folder_list.setCurrentRow(selected_row)
        self.folder_list.blockSignals(False)

        item = self.folder_list.currentItem()
        if item:
            self.current_kind = item.data(ROLE_KIND)
            self.current_folder_id = item.data(ROLE_FOLDER_ID)

    def current_rows(self):
        if self.current_kind == "folder" and self.current_folder_id is not None:
            return self.db.list_papers(folder_id=self.current_folder_id)
        if self.current_kind == "unfiled":
            return self.db.list_papers(unfiled=True)
        return self.db.list_papers()

    def refresh_table(self) -> None:
        rows = self.filter_rows_by_search(self.current_rows())

        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.table.setRowCount(len(rows))

        for row_index, row in enumerate(rows):
            values_by_key = {
                "id": row["id"],
                "title": row["title"],
                "creators": row["creators"],
                "year": row["year"],
                "date": row["date"],
                "publication": row["publication"],
                "journal_abbr": row["journal_abbr"],
                "volume": row["volume"],
                "pages": row["pages"],
                "doi": row["doi"],
                "url": row["url"],
                "date_added": row["date_added"],
                "pdf_path": "",
            }

            for col_index, (key, _) in enumerate(self.COLUMNS):
                value = values_by_key.get(key)
                item = QTableWidgetItem("" if value is None else str(value))

                if key == "id":
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                if key == "url" and value:
                    font = item.font()
                    font.setUnderline(True)
                    item.setFont(font)
                    item.setForeground(QBrush(QColor("blue")))
                    item.setToolTip("Click to open URL in browser")

                if key == "pdf_path" and row["pdf_path"]:
                    if not self.pdf_icon.isNull():
                        item.setIcon(self.pdf_icon)
                    item.setToolTip("PDF attached. Double-click row to open PDF.")

                item.setData(Qt.UserRole, row["id"])
                self.table.setItem(row_index, col_index, item)

        self.table.setSortingEnabled(True)
        self.apply_column_visibility()
        self.status_label.setText(
            self.table_status_message(len(rows))
        )

        self.clear_edit_panel()

    # ---------- Search ----------

    def search_query(self) -> str:
        box = getattr(self, "search_box", None)
        if box is None:
            return ""
        return box.text().strip().lower()

    def filter_rows_by_search(self, rows):
        query = self.search_query()
        if not query:
            return rows

        terms = [term for term in query.split() if term]
        if not terms:
            return rows

        filtered = []
        for row in rows:
            title = "" if row["title"] is None else str(row["title"])
            creators = "" if row["creators"] is None else str(row["creators"])
            haystack = f"{title} {creators}".lower()
            if all(term in haystack for term in terms):
                filtered.append(row)

        return filtered

    def on_search_text_changed(self, text: str) -> None:
        self.refresh_table()

    def focus_search_box(self) -> None:
        self.search_box.setFocus()
        self.search_box.selectAll()

    def clear_search_box(self) -> None:
        self.search_box.clear()

    def table_status_message(self, row_count: int) -> str:
        query = self.search_query()
        if query:
            return f"{row_count} papers found for '{query}'. Search targets Title and Creator."
        return f"{row_count} papers loaded. PDF icon indicates an attached PDF. URL column opens browser on click."

    # ---------- Folder logic ----------

    def on_folder_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if current is None:
            return

        self.current_kind = current.data(ROLE_KIND)
        self.current_folder_id = current.data(ROLE_FOLDER_ID)
        self.refresh_table()

    def create_folder_dialog(self) -> int | None:
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if not ok or not name.strip():
            return None

        try:
            folder_id = self.db.create_folder(name.strip())
        except Exception as e:
            QMessageBox.warning(self, "Create folder failed", str(e))
            return None

        self.current_kind = "folder"
        self.current_folder_id = folder_id
        self.refresh_folders()
        self.refresh_table()
        return folder_id

    def show_folder_context_menu(self, pos: QPoint) -> None:
        item = self.folder_list.itemAt(pos)
        menu = QMenu(self)

        new_action = menu.addAction("New Folder")
        copy_refs_action = menu.addAction("Copy APS Reference List")

        rename_action = None
        delete_action = None

        if item and item.data(ROLE_KIND) == "folder":
            menu.addSeparator()
            rename_action = menu.addAction("Rename Folder")
            delete_action = menu.addAction("Delete Folder")

        action = menu.exec(self.folder_list.mapToGlobal(pos))

        if action == new_action:
            self.create_folder_dialog()
        elif action == copy_refs_action:
            self.copy_current_folder_references()
        elif rename_action is not None and action == rename_action and item:
            self.rename_folder_dialog(item)
        elif delete_action is not None and action == delete_action and item:
            self.delete_folder_dialog(item)

    def rename_folder_dialog(self, item: QListWidgetItem) -> None:
        folder_id = item.data(ROLE_FOLDER_ID)
        old_name = item.text().rsplit(" (", 1)[0]

        new_name, ok = QInputDialog.getText(self, "Rename Folder", "New folder name:", text=old_name)
        if not ok or not new_name.strip():
            return

        try:
            self.db.rename_folder(folder_id, new_name.strip())
        except Exception as e:
            QMessageBox.warning(self, "Rename failed", str(e))
            return

        self.refresh_folders()

    def delete_folder_dialog(self, item: QListWidgetItem) -> None:
        folder_id = item.data(ROLE_FOLDER_ID)
        name = item.text().rsplit(" (", 1)[0]

        reply = QMessageBox.question(
            self,
            "Delete Folder",
            f"Delete folder '{name}'?\n\nPapers themselves will not be deleted.",
        )

        if reply != QMessageBox.Yes:
            return

        self.db.delete_folder(folder_id)
        self.current_kind = "all"
        self.current_folder_id = None
        self.refresh_folders()
        self.refresh_table()

    # ---------- Import ----------

    def current_import_folder_id(self) -> int | None:
        if self.current_kind == "folder" and self.current_folder_id is not None:
            return int(self.current_folder_id)
        return None

    def import_pdf_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import PDF",
            str(Path.home()),
            "PDF Files (*.pdf)",
        )

        if not paths:
            return

        self.import_paths([Path(p) for p in paths])

    def import_bibtex_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import BibTeX",
            str(Path.home()),
            "BibTeX Files (*.bib *.bibtex)",
        )

        if not paths:
            return

        self.import_paths([Path(p) for p in paths])

    def import_paths(self, paths: list[Path]) -> None:
        imported_pdf = 0
        imported_bib_entries = 0
        errors: list[str] = []
        folder_id = self.current_import_folder_id()

        for path in paths:
            try:
                suffix = path.suffix.lower()

                if suffix == ".pdf":
                    record = import_pdf_file(path, PDF_DIR)

                    # DOIがPDFから抽出できた場合、doi.orgからBibTeXを取得し、
                    # title, authors, journal, volume, pages, yearなどの空欄を補完する。
                    bibtex_record = None
                    if record.get("doi"):
                        bibtex_record = fetch_bibtex_record_by_doi(record["doi"])
                    record = merge_pdf_record_with_bibtex(record, bibtex_record)

                    self.db.upsert_pdf_stub(record, folder_id=folder_id)
                    imported_pdf += 1

                elif suffix in BIBTEX_SUFFIXES:
                    records = import_bibtex_file(path)
                    for record in records:
                        if record.get("publication") and not record.get("journal_abbr"):
                            record["journal_abbr"] = infer_journal_abbr(record.get("publication"))
                        self.db.upsert_bibtex_record(record, folder_id=folder_id)
                    imported_bib_entries += len(records)

                else:
                    errors.append(f"Unsupported file: {path}")

            except Exception as e:
                errors.append(f"{path.name}: {e}")

        self.full_refresh()

        message = f"Imported PDFs: {imported_pdf}\nImported BibTeX entries: {imported_bib_entries}"
        if folder_id is not None:
            message += "\nImported into the selected folder."
        if errors:
            message += "\n\nErrors:\n" + "\n".join(errors[:10])

        QMessageBox.information(self, "Import finished", message)

    # ---------- Paper selection / editing ----------

    def selected_paper_ids(self) -> list[int]:
        ids = set()
        for item in self.table.selectedItems():
            row = item.row()
            id_item = self.table.item(row, 0)
            if id_item is None:
                continue
            try:
                ids.add(int(id_item.text()))
            except ValueError:
                continue
        return sorted(ids)

    def selected_paper_id(self) -> int | None:
        ids = self.selected_paper_ids()
        if not ids:
            return None
        return ids[0]

    def on_table_selection_changed(self) -> None:
        ids = self.selected_paper_ids()
        if len(ids) != 1:
            self.clear_edit_panel()
            if len(ids) > 1:
                self.status_label.setText(f"{len(ids)} papers selected. Right-click to move, delete, or copy references.")
            return

        self.load_paper_into_edit_panel(ids[0])

    def load_paper_into_edit_panel(self, paper_id: int) -> None:
        row = self.db.get_paper(paper_id)
        if row is None:
            self.clear_edit_panel()
            return

        self.loading_edit_panel = True
        self.current_paper_id = paper_id

        for key, _, _ in self.EDIT_FIELDS:
            widget = self.edit_widgets[key]
            value = row[key] if key in row.keys() else ""
            value = "" if value is None else str(value)

            if isinstance(widget, QPlainTextEdit):
                widget.setPlainText(value)
            else:
                widget.setText(value)

        self.loading_edit_panel = False
        self.set_edit_panel_enabled(True)

    def clear_edit_panel(self) -> None:
        self.loading_edit_panel = True
        self.current_paper_id = None

        for widget in self.edit_widgets.values():
            if isinstance(widget, QPlainTextEdit):
                widget.setPlainText("")
            else:
                widget.setText("")

        self.loading_edit_panel = False
        self.set_edit_panel_enabled(False)

    def set_edit_panel_enabled(self, enabled: bool) -> None:
        for widget in self.edit_widgets.values():
            widget.setEnabled(enabled)

        self.copy_ref_button.setEnabled(enabled)
        self.open_url_button.setEnabled(enabled)

    def on_edit_text_edited(self) -> None:
        if self.loading_edit_panel:
            return

        sender = self.sender()

        # DOI入力中はURLを自動更新
        doi_widget = self.edit_widgets.get("doi")
        url_widget = self.edit_widgets.get("url")
        if sender is doi_widget and isinstance(doi_widget, QLineEdit) and isinstance(url_widget, QLineEdit):
            doi = normalize_doi(doi_widget.text())
            if doi:
                url_widget.setText(doi_to_url(doi) or "")

        # Publication入力中はJournal Abbrを自動補完
        pub_widget = self.edit_widgets.get("publication")
        abbr_widget = self.edit_widgets.get("journal_abbr")
        if sender is pub_widget and isinstance(pub_widget, QLineEdit) and isinstance(abbr_widget, QLineEdit):
            abbr = infer_journal_abbr(pub_widget.text())
            if abbr:
                current = abbr_widget.text().strip()
                # 空欄、または以前の自動略称に近い場合は上書き。手入力がありそうな場合は空欄時のみ。
                if not current:
                    abbr_widget.setText(abbr)

        self.schedule_autosave()

    def schedule_autosave(self) -> None:
        if self.loading_edit_panel or self.current_paper_id is None:
            return
        self.autosave_timer.start(600)

    def collect_edit_updates(self) -> dict:
        updates = {}
        for key, _, _ in self.EDIT_FIELDS:
            widget = self.edit_widgets[key]
            if isinstance(widget, QPlainTextEdit):
                updates[key] = widget.toPlainText()
            else:
                updates[key] = widget.text()

        # DOIとURLの整合
        doi = normalize_doi(updates.get("doi"))
        if doi:
            updates["doi"] = doi
            if not updates.get("url"):
                updates["url"] = doi_to_url(doi)

        # PublicationからJournal Abbrを補完
        if updates.get("publication") and not updates.get("journal_abbr"):
            inferred = infer_journal_abbr(updates.get("publication"))
            if inferred:
                updates["journal_abbr"] = inferred
                abbr_widget = self.edit_widgets.get("journal_abbr")
                if isinstance(abbr_widget, QLineEdit):
                    abbr_widget.setText(inferred)

        return updates

    def autosave_current_paper(self) -> None:
        if self.loading_edit_panel or self.current_paper_id is None:
            return

        paper_id = self.current_paper_id
        updates = self.collect_edit_updates()

        try:
            self.db.update_paper(paper_id, updates)
        except Exception as e:
            self.status_label.setText(f"Auto-save failed: {e}")
            return

        self.refresh_table_preserve_selection(paper_id)
        self.status_label.setText("Changes saved.")

    def refresh_table_preserve_selection(self, paper_id: int | None = None) -> None:
        if paper_id is None:
            paper_id = self.current_paper_id

        self.table.blockSignals(True)
        self.refresh_folders()
        self.refresh_table()
        self.table.blockSignals(False)

        if paper_id is not None:
            self.select_paper_by_id(paper_id)
            self.load_paper_into_edit_panel(paper_id)

    def select_paper_by_id(self, paper_id: int | None) -> None:
        if paper_id is None:
            return

        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == str(paper_id):
                self.table.selectRow(row)
                return

    # ---------- Context menus ----------

    def show_paper_context_menu(self, pos: QPoint) -> None:
        paper_ids = self.selected_paper_ids()
        if not paper_ids:
            return

        n = len(paper_ids)
        menu = QMenu(self)

        open_action = None
        if n == 1:
            open_action = menu.addAction("Open PDF")

        copy_ref_action = menu.addAction("Copy APS Reference" if n == 1 else f"Copy APS References ({n})")
        fetch_metadata_action = menu.addAction("Fetch BibTeX by DOI" if n == 1 else f"Fetch BibTeX by DOI ({n})")
        menu.addSeparator()

        move_menu = menu.addMenu("Move to Folder")
        folder_actions: dict[QAction, int] = {}

        for folder in self.db.list_folders():
            action = move_menu.addAction(folder["name"])
            folder_actions[action] = int(folder["id"])

        if folder_actions:
            move_menu.addSeparator()

        new_folder_action = move_menu.addAction("New Folder...")

        remove_from_current_action = None
        if self.current_kind == "folder" and self.current_folder_id is not None:
            remove_from_current_action = menu.addAction("Remove from Current Folder" if n == 1 else f"Remove {n} Papers from Current Folder")

        menu.addSeparator()
        delete_action = menu.addAction("Delete Paper from Library" if n == 1 else f"Delete {n} Papers from Library")

        action = menu.exec(self.table.mapToGlobal(pos))

        if open_action is not None and action == open_action:
            self.open_selected_pdf()
        elif action == copy_ref_action:
            self.copy_selected_paper_reference()
        elif action == fetch_metadata_action:
            self.fetch_metadata_for_selected_papers()
        elif action in folder_actions:
            self.move_selected_papers_to_folder(folder_actions[action])
        elif action == new_folder_action:
            folder_id = self.create_folder_dialog()
            if folder_id is not None:
                self.move_selected_papers_to_folder(folder_id)
        elif remove_from_current_action is not None and action == remove_from_current_action:
            self.remove_selected_papers_from_current_folder()
        elif action == delete_action:
            self.delete_selected_papers()

    def move_selected_papers_to_folder(self, folder_id: int) -> None:
        paper_ids = self.selected_paper_ids()
        if not paper_ids:
            return

        self.db.move_papers_to_folder(paper_ids, folder_id)
        self.current_kind = "folder"
        self.current_folder_id = folder_id
        self.full_refresh()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and int(item.text()) in paper_ids:
                self.table.selectRow(row)

    def remove_selected_papers_from_current_folder(self) -> None:
        paper_ids = self.selected_paper_ids()
        if not paper_ids or self.current_folder_id is None:
            return

        self.db.remove_papers_from_folder(paper_ids, self.current_folder_id)
        self.full_refresh()

    def delete_selected_papers(self) -> None:
        paper_ids = self.selected_paper_ids()
        if not paper_ids:
            return

        reply = QMessageBox.question(
            self,
            "Delete Papers",
            f"Delete {len(paper_ids)} paper(s) from the library?\n\nThe PDF files themselves will not be deleted.",
        )

        if reply != QMessageBox.Yes:
            return

        self.db.delete_papers(paper_ids)
        self.full_refresh()

    def set_references_to_clipboard(self, plain_text: str, html_text: str) -> None:
        mime = QMimeData()
        mime.setText(plain_text)
        mime.setHtml(html_text)
        QApplication.clipboard().setMimeData(mime)

    # ---------- DOI metadata fetch ----------

    def fetch_metadata_for_selected_papers(self) -> None:
        paper_ids = self.selected_paper_ids()
        if not paper_ids and self.current_paper_id is not None:
            paper_ids = [self.current_paper_id]

        if not paper_ids:
            QMessageBox.information(self, "No paper selected", "Select one or more papers first.")
            return

        updated = 0
        skipped_no_doi = 0
        failed = 0

        for paper_id in paper_ids:
            row = self.db.get_paper(paper_id)
            if row is None:
                continue

            doi = row["doi"] if "doi" in row.keys() else None
            if not doi:
                skipped_no_doi += 1
                continue

            bibtex_record = fetch_bibtex_record_by_doi(doi)
            if not bibtex_record:
                failed += 1
                continue

            # 既存DBの内容を基準に、空欄だけBibTeXで補完する。
            current = {key: row[key] for key in row.keys()}
            merged = merge_pdf_record_with_bibtex(current, bibtex_record)

            # PDF取り込み時と違い、手入力済みのtitleは尊重する。
            if current.get("title"):
                merged["title"] = current.get("title")

            updates = {
                "title": merged.get("title"),
                "creators": merged.get("creators"),
                "year": merged.get("year"),
                "date": merged.get("date"),
                "publication": merged.get("publication"),
                "journal_abbr": merged.get("journal_abbr"),
                "doi": merged.get("doi"),
                "url": merged.get("url"),
                "bibtex_key": merged.get("bibtex_key"),
                "bibtex_raw": merged.get("bibtex_raw"),
                "notes": merged.get("notes"),
                "volume": merged.get("volume"),
                "pages": merged.get("pages"),
                "issue": merged.get("issue"),
                "publisher": merged.get("publisher"),
                "entry_type": merged.get("entry_type"),
            }
            self.db.update_paper(paper_id, updates)
            updated += 1

        self.full_refresh()

        QMessageBox.information(
            self,
            "Fetch BibTeX by DOI",
            f"Updated: {updated}\nSkipped, no DOI: {skipped_no_doi}\nFailed: {failed}",
        )

    # ---------- Reference copy ----------

    def copy_selected_paper_reference(self) -> None:
        paper_ids = self.selected_paper_ids()
        if not paper_ids and self.current_paper_id is not None:
            paper_ids = [self.current_paper_id]
        if not paper_ids:
            return

        rows = self.db.get_papers_by_ids(paper_ids)
        if not rows:
            return

        if len(rows) == 1:
            text = format_aps_reference(rows[0])
            html_text = f"<html><body>{format_aps_reference_html(rows[0])}</body></html>"
        else:
            text = format_aps_references(rows)
            html_text = format_aps_references_html(rows)

        self.set_references_to_clipboard(text, html_text)
        self.status_label.setText(f"{len(rows)} APS reference(s) copied to clipboard. Volume is bold in rich-text paste.")

    def copy_current_folder_references(self) -> None:
        rows = self.current_rows()

        if not rows:
            QMessageBox.information(self, "No papers", "There are no papers to export.")
            return

        text = format_aps_references(rows)
        html_text = format_aps_references_html(rows)
        self.set_references_to_clipboard(text, html_text)
        self.status_label.setText(f"{len(rows)} APS references copied to clipboard. Volume is bold in rich-text paste.")
        QMessageBox.information(self, "Copied", f"{len(rows)} APS references were copied to the clipboard.")

    # ---------- URL and PDF open ----------

    def url_column_index(self) -> int:
        for idx, (key, _) in enumerate(self.COLUMNS):
            if key == "url":
                return idx
        return -1

    def on_cell_clicked(self, row: int, column: int) -> None:
        if column == self.url_column_index():
            item = self.table.item(row, column)
            if item and item.text().strip():
                open_url(item.text().strip())

    def on_cell_double_clicked(self, row: int, column: int) -> None:
        if column == self.url_column_index():
            return
        self.open_selected_pdf()

    def open_selected_pdf(self) -> None:
        paper_id = self.selected_paper_id()
        if paper_id is None:
            return

        pdf_path = self.db.get_pdf_path(paper_id)

        if not pdf_path:
            QMessageBox.warning(self, "No PDF", "This paper does not have a PDF path.")
            return

        path = Path(pdf_path)

        if not path.exists():
            QMessageBox.warning(self, "PDF not found", f"PDF file does not exist:\n{pdf_path}")
            return

        open_file_with_default_app(path)

    def open_current_url(self) -> None:
        url_widget = self.edit_widgets.get("url")
        if isinstance(url_widget, QLineEdit):
            url = url_widget.text().strip()
            if url:
                open_url(url)

    # ---------- Drag and drop ----------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in ({".pdf"} | BIBTEX_SUFFIXES):
                    event.acceptProposedAction()
                    return

        event.ignore()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        paths = [p for p in paths if p.suffix.lower() in ({".pdf"} | BIBTEX_SUFFIXES)]

        if paths:
            self.import_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


def open_file_with_default_app(path: Path) -> None:
    system = platform.system()

    if system == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def open_url(url: str) -> None:
    url = url.strip()
    if not url:
        return
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    webbrowser.open(url)


def main() -> int:
    app = QApplication(sys.argv)
    window = PaperManagerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
