from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


BASE_PAPER_COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "title": "TEXT",
    "creators": "TEXT",
    "year": "TEXT",
    "date": "TEXT",
    "publication": "TEXT",
    "journal_abbr": "TEXT",
    "doi": "TEXT UNIQUE",
    "url": "TEXT",
    "date_added": "TEXT NOT NULL",
    "pdf_path": "TEXT",
    "bibtex_key": "TEXT",
    "bibtex_raw": "TEXT",
    "file_hash": "TEXT UNIQUE",
    "notes": "TEXT",
    "volume": "TEXT",
    "pages": "TEXT",
    "issue": "TEXT",
    "publisher": "TEXT",
    "entry_type": "TEXT",
}


PAPER_UPDATE_COLUMNS = [
    "title",
    "creators",
    "year",
    "date",
    "publication",
    "journal_abbr",
    "doi",
    "url",
    "pdf_path",
    "bibtex_key",
    "bibtex_raw",
    "file_hash",
    "notes",
    "volume",
    "pages",
    "issue",
    "publisher",
    "entry_type",
]


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                creators TEXT,
                year TEXT,
                date TEXT,
                publication TEXT,
                journal_abbr TEXT,
                doi TEXT UNIQUE,
                url TEXT,
                date_added TEXT NOT NULL,
                pdf_path TEXT,
                bibtex_key TEXT,
                bibtex_raw TEXT,
                file_hash TEXT UNIQUE,
                notes TEXT,
                volume TEXT,
                pages TEXT,
                issue TEXT,
                publisher TEXT,
                entry_type TEXT
            );
            """
        )

        self.ensure_paper_columns()

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_folders (
                paper_id INTEGER NOT NULL,
                folder_id INTEGER NOT NULL,
                PRIMARY KEY (paper_id, folder_id),
                FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
                FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE CASCADE
            );
            """
        )

        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_folders_folder ON paper_folders(folder_id);")
        self.conn.commit()
        self.backfill_urls_from_doi()

        # Existing duplicate records are also consolidated automatically at startup.
        self.consolidate_duplicates()

    def ensure_paper_columns(self) -> None:
        cur = self.conn.execute("PRAGMA table_info(papers);")
        existing = {row["name"] for row in cur.fetchall()}

        for name, col_type in BASE_PAPER_COLUMNS.items():
            if name in existing or name == "id":
                continue
            plain_type = col_type.replace(" UNIQUE", "").replace(" NOT NULL", "")
            self.conn.execute(f"ALTER TABLE papers ADD COLUMN {name} {plain_type};")

        self.conn.commit()

    def backfill_urls_from_doi(self) -> None:
        cur = self.conn.execute("SELECT id, doi, url FROM papers;")
        for row in cur.fetchall():
            doi = normalize_doi(row["doi"])
            url = empty_to_none(row["url"])
            if doi and not url:
                self.conn.execute(
                    "UPDATE papers SET doi = ?, url = ? WHERE id = ?;",
                    (doi, doi_to_url(doi), row["id"]),
                )
        self.conn.commit()

    # ---------- Folders ----------

    def list_folders(self) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            """
            SELECT
                f.id,
                f.name,
                COUNT(pf.paper_id) AS paper_count
            FROM folders f
            LEFT JOIN paper_folders pf ON pf.folder_id = f.id
            GROUP BY f.id, f.name
            ORDER BY LOWER(f.name) ASC;
            """
        )
        return list(cur.fetchall())

    def create_folder(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Folder name is empty.")

        cur = self.conn.execute(
            "INSERT INTO folders(name) VALUES (?);",
            (name,),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def rename_folder(self, folder_id: int, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Folder name is empty.")

        self.conn.execute(
            "UPDATE folders SET name = ? WHERE id = ?;",
            (new_name, folder_id),
        )
        self.conn.commit()

    def delete_folder(self, folder_id: int) -> None:
        self.conn.execute("DELETE FROM paper_folders WHERE folder_id = ?;", (folder_id,))
        self.conn.execute("DELETE FROM folders WHERE id = ?;", (folder_id,))
        self.conn.commit()

    def assign_paper_to_folder(self, paper_id: int, folder_id: int) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO paper_folders(paper_id, folder_id)
            VALUES (?, ?);
            """,
            (paper_id, folder_id),
        )
        self.conn.commit()

    def move_paper_to_folder(self, paper_id: int, folder_id: int) -> None:
        self.conn.execute("DELETE FROM paper_folders WHERE paper_id = ?;", (paper_id,))
        self.conn.execute(
            """
            INSERT OR IGNORE INTO paper_folders(paper_id, folder_id)
            VALUES (?, ?);
            """,
            (paper_id, folder_id),
        )
        self.conn.commit()

    def move_papers_to_folder(self, paper_ids: list[int], folder_id: int) -> None:
        for paper_id in paper_ids:
            self.conn.execute("DELETE FROM paper_folders WHERE paper_id = ?;", (paper_id,))
            self.conn.execute(
                """
                INSERT OR IGNORE INTO paper_folders(paper_id, folder_id)
                VALUES (?, ?);
                """,
                (paper_id, folder_id),
            )
        self.conn.commit()

    def remove_paper_from_folder(self, paper_id: int, folder_id: int) -> None:
        self.conn.execute(
            "DELETE FROM paper_folders WHERE paper_id = ? AND folder_id = ?;",
            (paper_id, folder_id),
        )
        self.conn.commit()

    def remove_papers_from_folder(self, paper_ids: list[int], folder_id: int) -> None:
        for paper_id in paper_ids:
            self.conn.execute(
                "DELETE FROM paper_folders WHERE paper_id = ? AND folder_id = ?;",
                (paper_id, folder_id),
            )
        self.conn.commit()

    # ---------- Papers ----------

    def list_papers(self, folder_id: int | None = None, unfiled: bool = False) -> list[sqlite3.Row]:
        if folder_id is not None:
            cur = self.conn.execute(
                """
                SELECT p.*
                FROM papers p
                INNER JOIN paper_folders pf ON pf.paper_id = p.id
                WHERE pf.folder_id = ?
                ORDER BY p.date_added DESC, p.id DESC;
                """,
                (folder_id,),
            )
            return list(cur.fetchall())

        if unfiled:
            cur = self.conn.execute(
                """
                SELECT p.*
                FROM papers p
                LEFT JOIN paper_folders pf ON pf.paper_id = p.id
                WHERE pf.folder_id IS NULL
                ORDER BY p.date_added DESC, p.id DESC;
                """
            )
            return list(cur.fetchall())

        cur = self.conn.execute(
            """
            SELECT *
            FROM papers
            ORDER BY date_added DESC, id DESC;
            """
        )
        return list(cur.fetchall())

    def get_paper(self, paper_id: int) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM papers WHERE id = ?;", (paper_id,))
        return cur.fetchone()

    def get_papers_by_ids(self, paper_ids: list[int]) -> list[sqlite3.Row]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        cur = self.conn.execute(
            f"SELECT * FROM papers WHERE id IN ({placeholders});",
            paper_ids,
        )
        rows = list(cur.fetchall())
        order = {pid: i for i, pid in enumerate(paper_ids)}
        rows.sort(key=lambda r: order.get(int(r["id"]), 10**9))
        return rows

    def get_pdf_path(self, paper_id: int) -> str | None:
        row = self.get_paper(paper_id)
        if row is None:
            return None
        return row["pdf_path"]

    def update_paper(self, paper_id: int, updates: dict[str, Any]) -> None:
        allowed = set(PAPER_UPDATE_COLUMNS)
        updates = {k: empty_to_none(v) for k, v in updates.items() if k in allowed}

        if "doi" in updates:
            updates["doi"] = normalize_doi(updates["doi"])

        doi = updates.get("doi")
        if doi and not updates.get("url"):
            updates["url"] = doi_to_url(doi)

        if not updates:
            return

        # If manual editing makes this paper duplicate an existing one,
        # merge into the existing record instead of creating a conflict.
        duplicate_id = self.find_existing_paper_id(updates, exclude_id=paper_id)
        if duplicate_id is not None:
            safe_updates = {k: v for k, v in updates.items() if k not in {"doi", "file_hash"}}
            if safe_updates:
                self._apply_update(paper_id, safe_updates)
            self.merge_paper_records(duplicate_id, paper_id)
            return

        self._apply_update(paper_id, updates)
        self.merge_duplicate_for_paper(paper_id)

    def _apply_update(self, paper_id: int, updates: dict[str, Any]) -> None:
        updates = {k: v for k, v in updates.items() if k in PAPER_UPDATE_COLUMNS}

        if not updates:
            return

        # Avoid UNIQUE conflicts by merging conflicting records first.
        for key in ("doi", "file_hash"):
            value = updates.get(key)
            if not value:
                continue
            conflict_id = self.find_id_by_field(key, value, exclude_id=paper_id)
            if conflict_id is not None:
                self.merge_paper_records(paper_id, conflict_id)

        assignments = ", ".join(f"{key} = ?" for key in updates.keys())
        values = list(updates.values()) + [paper_id]
        self.conn.execute(f"UPDATE papers SET {assignments} WHERE id = ?;", values)
        self.conn.commit()

    def delete_paper(self, paper_id: int) -> None:
        self.delete_papers([paper_id])

    def delete_papers(self, paper_ids: list[int]) -> None:
        for paper_id in paper_ids:
            self.conn.execute("DELETE FROM paper_folders WHERE paper_id = ?;", (paper_id,))
            self.conn.execute("DELETE FROM papers WHERE id = ?;", (paper_id,))
        self.conn.commit()

    def upsert_pdf_stub(self, record: dict[str, Any], folder_id: int | None = None) -> int:
        record = self.prepare_record(record)

        existing_id = self.find_existing_paper_id(record, include_bibtex_key=False)
        if existing_id is None and record.get("file_hash"):
            existing_id = self.find_id_by_field("file_hash", record.get("file_hash"))

        if existing_id is not None:
            self.update_existing_from_record(existing_id, record, prefer_new=True)
            if folder_id is not None:
                self.assign_paper_to_folder(existing_id, folder_id)
            return existing_id

        cur = self.conn.execute(
            """
            INSERT INTO papers (
                title, creators, year, date, publication, journal_abbr,
                doi, url, date_added, pdf_path, bibtex_key, bibtex_raw, file_hash,
                notes, volume, pages, issue, publisher, entry_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.get("title"),
                record.get("creators"),
                record.get("year"),
                record.get("date"),
                record.get("publication"),
                record.get("journal_abbr"),
                record.get("doi"),
                record.get("url"),
                record["date_added"],
                record.get("pdf_path"),
                record.get("bibtex_key"),
                record.get("bibtex_raw"),
                record.get("file_hash"),
                record.get("notes"),
                record.get("volume"),
                record.get("pages"),
                record.get("issue"),
                record.get("publisher"),
                record.get("entry_type"),
            ),
        )
        self.conn.commit()
        paper_id = int(cur.lastrowid)

        # In case a race or title variant made a duplicate, consolidate once.
        paper_id = self.merge_duplicate_for_paper(paper_id)

        if folder_id is not None:
            self.assign_paper_to_folder(paper_id, folder_id)
        return paper_id

    def upsert_bibtex_record(self, record: dict[str, Any], folder_id: int | None = None) -> int:
        record = self.prepare_record(record)

        existing_id = self.find_existing_paper_id(record, include_bibtex_key=True)

        if existing_id is not None:
            self.update_existing_from_record(existing_id, record, prefer_new=True)
            if folder_id is not None:
                self.assign_paper_to_folder(existing_id, folder_id)
            return existing_id

        cur = self.conn.execute(
            """
            INSERT INTO papers (
                title, creators, year, date, publication, journal_abbr,
                doi, url, date_added, pdf_path, bibtex_key, bibtex_raw, file_hash,
                notes, volume, pages, issue, publisher, entry_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.get("title"),
                record.get("creators"),
                record.get("year"),
                record.get("date"),
                record.get("publication"),
                record.get("journal_abbr"),
                record.get("doi"),
                record.get("url"),
                record["date_added"],
                record.get("pdf_path"),
                record.get("bibtex_key"),
                record.get("bibtex_raw"),
                record.get("file_hash"),
                record.get("notes"),
                record.get("volume"),
                record.get("pages"),
                record.get("issue"),
                record.get("publisher"),
                record.get("entry_type"),
            ),
        )
        self.conn.commit()
        paper_id = int(cur.lastrowid)

        paper_id = self.merge_duplicate_for_paper(paper_id)

        if folder_id is not None:
            self.assign_paper_to_folder(paper_id, folder_id)
        return paper_id

    # ---------- Duplicate handling ----------

    def prepare_record(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {k: empty_to_none(v) for k, v in dict(record).items()}
        record["doi"] = normalize_doi(record.get("doi"))

        if record.get("doi") and not record.get("url"):
            record["url"] = doi_to_url(record["doi"])

        return record

    def find_existing_paper_id(
        self,
        record: dict[str, Any],
        exclude_id: int | None = None,
        include_bibtex_key: bool = False,
    ) -> int | None:
        doi = normalize_doi(record.get("doi"))
        if doi:
            found = self.find_id_by_field("doi", doi, exclude_id=exclude_id)
            if found is not None:
                return found

        title_key = normalize_title_for_match(record.get("title"))
        if title_key:
            cur = self.conn.execute("SELECT id, title FROM papers ORDER BY id ASC;")
            for row in cur.fetchall():
                row_id = int(row["id"])
                if exclude_id is not None and row_id == int(exclude_id):
                    continue
                if normalize_title_for_match(row["title"]) == title_key:
                    return row_id

        if include_bibtex_key:
            bibtex_key = empty_to_none(record.get("bibtex_key"))
            if bibtex_key:
                found = self.find_id_by_field("bibtex_key", bibtex_key, exclude_id=exclude_id)
                if found is not None:
                    return found

        return None

    def find_id_by_field(self, field: str, value: Any, exclude_id: int | None = None) -> int | None:
        if field not in {"doi", "file_hash", "bibtex_key"}:
            raise ValueError(f"Unsupported lookup field: {field}")

        value = normalize_doi(value) if field == "doi" else empty_to_none(value)
        if not value:
            return None

        if exclude_id is None:
            cur = self.conn.execute(f"SELECT id FROM papers WHERE {field} = ? ORDER BY id ASC LIMIT 1;", (value,))
        else:
            cur = self.conn.execute(
                f"SELECT id FROM papers WHERE {field} = ? AND id <> ? ORDER BY id ASC LIMIT 1;",
                (value, int(exclude_id)),
            )
        row = cur.fetchone()
        return int(row["id"]) if row else None

    def update_existing_from_record(self, paper_id: int, record: dict[str, Any], prefer_new: bool = True) -> None:
        record = self.prepare_record(record)
        current = self.get_paper(paper_id)
        if current is None:
            return

        updates: dict[str, Any] = {}

        for key in PAPER_UPDATE_COLUMNS:
            value = empty_to_none(record.get(key))
            if value is None:
                continue

            if key == "doi":
                value = normalize_doi(value)

            if key in {"doi", "file_hash"}:
                if not value:
                    continue
                conflict_id = self.find_id_by_field(key, value, exclude_id=paper_id)
                if conflict_id is not None:
                    self.merge_paper_records(paper_id, conflict_id)
                    current = self.get_paper(paper_id)
                    if current is None:
                        return

                current_value = empty_to_none(current[key])
                if current_value and str(current_value) != str(value):
                    # Keep the existing unique identifier if it differs.
                    continue

            current_value = empty_to_none(current[key])
            if prefer_new:
                # Prefer new bibliographic metadata, but avoid erasing existing nonempty values.
                updates[key] = value
            elif not current_value:
                updates[key] = value

        if updates:
            self._apply_update(paper_id, updates)

        self.merge_duplicate_for_paper(paper_id)

    def merge_duplicate_for_paper(self, paper_id: int) -> int:
        row = self.get_paper(paper_id)
        if row is None:
            return paper_id

        record = {key: row[key] for key in row.keys()}
        duplicate_id = self.find_existing_paper_id(record, exclude_id=paper_id)

        if duplicate_id is None:
            return paper_id

        target_id = min(int(paper_id), int(duplicate_id))
        source_id = max(int(paper_id), int(duplicate_id))
        self.merge_paper_records(target_id, source_id)
        return target_id

    def merge_paper_records(self, target_id: int, source_id: int) -> int:
        if target_id == source_id:
            return target_id

        target = self.get_paper(target_id)
        source = self.get_paper(source_id)
        if target is None or source is None:
            return target_id

        # Preserve folder assignments.
        cur = self.conn.execute("SELECT folder_id FROM paper_folders WHERE paper_id = ?;", (source_id,))
        for row in cur.fetchall():
            self.conn.execute(
                "INSERT OR IGNORE INTO paper_folders(paper_id, folder_id) VALUES (?, ?);",
                (target_id, row["folder_id"]),
            )

        updates: dict[str, Any] = {}

        for key in PAPER_UPDATE_COLUMNS:
            target_value = empty_to_none(target[key])
            source_value = empty_to_none(source[key])

            if source_value is None:
                continue

            if key in {"doi", "file_hash"}:
                if not target_value:
                    # Clear source unique value before assigning it to target.
                    self.conn.execute(f"UPDATE papers SET {key} = NULL WHERE id = ?;", (source_id,))
                    updates[key] = source_value
                continue

            if not target_value:
                updates[key] = source_value

        if updates:
            assignments = ", ".join(f"{key} = ?" for key in updates.keys())
            values = list(updates.values()) + [target_id]
            self.conn.execute(f"UPDATE papers SET {assignments} WHERE id = ?;", values)

        self.conn.execute("DELETE FROM paper_folders WHERE paper_id = ?;", (source_id,))
        self.conn.execute("DELETE FROM papers WHERE id = ?;", (source_id,))
        self.conn.commit()
        return target_id

    def consolidate_duplicates(self) -> int:
        merged_count = 0

        while True:
            rows = list(self.conn.execute("SELECT id, doi, title FROM papers ORDER BY id ASC;").fetchall())
            doi_seen: dict[str, int] = {}
            title_seen: dict[str, int] = {}
            merged_this_round = False

            for row in rows:
                paper_id = int(row["id"])

                doi = normalize_doi(row["doi"])
                if doi:
                    if doi in doi_seen:
                        self.merge_paper_records(doi_seen[doi], paper_id)
                        merged_count += 1
                        merged_this_round = True
                        break
                    doi_seen[doi] = paper_id

                title_key = normalize_title_for_match(row["title"])
                if title_key:
                    if title_key in title_seen:
                        self.merge_paper_records(title_seen[title_key], paper_id)
                        merged_count += 1
                        merged_this_round = True
                        break
                    title_seen[title_key] = paper_id

            if not merged_this_round:
                break

        return merged_count


def normalize_doi(value: Any) -> str | None:
    if value is None:
        return None

    doi = str(value).strip()
    if not doi:
        return None

    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://doi.org/", "")
    doi = doi.replace("doi:", "")
    doi = doi.strip().strip(".;,")
    doi = doi.lower()

    return doi or None


def doi_to_url(doi: str | None) -> str | None:
    doi = normalize_doi(doi)
    if not doi:
        return None
    return f"https://doi.org/{doi}"


def normalize_title_for_match(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    # Remove common BibTeX/LaTeX wrappers and normalize punctuation/spacing.
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("\\", " ")
    text = re.sub(r"[\u2018\u2019\u201c\u201d\"'`]", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    text = " ".join(text.split())

    # Avoid merging generic or extremely short titles.
    if len(text) < 8:
        return None

    generic_titles = {
        "untitled",
        "untitled document",
        "microsoft word",
        "full text",
        "article",
        "paper",
    }
    if text in generic_titles:
        return None

    return text


def empty_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    return value
