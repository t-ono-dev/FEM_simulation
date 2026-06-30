from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase


def import_bibtex_file(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()

    with path.open("r", encoding="utf-8") as f:
        db = bibtexparser.load(f)

    records: list[dict[str, Any]] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for entry in db.entries:
        raw = entry_to_bibtex(entry)

        title = clean_bibtex_text(entry.get("title"))
        authors = clean_bibtex_text(entry.get("author"))
        year = entry.get("year")
        date = entry.get("date") or year
        publication = (
            entry.get("journal")
            or entry.get("journaltitle")
            or entry.get("booktitle")
            or entry.get("publisher")
        )
        journal_abbr = (
            entry.get("shortjournal")
            or entry.get("journalabbr")
            or entry.get("abbr")
        )

        pdf_path = extract_file_path(entry.get("file"))

        records.append(
            {
                "title": title,
                "creators": authors,
                "year": year,
                "date": date,
                "publication": clean_bibtex_text(publication),
                "journal_abbr": clean_bibtex_text(journal_abbr),
                "doi": entry.get("doi"),
                "url": entry.get("url"),
                "date_added": now,
                "pdf_path": pdf_path,
                "bibtex_key": entry.get("ID"),
                "bibtex_raw": raw,
                "file_hash": None,
                "notes": clean_bibtex_text(entry.get("note")),
                "volume": clean_bibtex_text(entry.get("volume")),
                "pages": clean_bibtex_text(entry.get("pages")),
                "issue": clean_bibtex_text(entry.get("number") or entry.get("issue")),
                "publisher": clean_bibtex_text(entry.get("publisher")),
                "entry_type": clean_bibtex_text(entry.get("ENTRYTYPE")),
            }
        )

    return records


def entry_to_bibtex(entry: dict[str, Any]) -> str:
    db = BibDatabase()
    db.entries = [entry]

    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None

    return bibtexparser.dumps(db, writer=writer).strip()


def clean_bibtex_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\&", "&")
    text = text.replace("--", "-")
    text = " ".join(text.split())
    return text or None


def extract_file_path(value: Any) -> str | None:
    if not value:
        return None

    text = str(value)
    candidates = text.split(";")

    for candidate in candidates:
        joined = candidate

        if ".pdf" not in joined.lower():
            continue

        cleaned = joined.replace("\\:", ":")
        cleaned = cleaned.strip()

        while cleaned.startswith(":"):
            cleaned = cleaned[1:]

        lower = cleaned.lower()
        idx = lower.find(".pdf")
        if idx >= 0:
            cleaned = cleaned[: idx + 4]

        p = Path(cleaned)
        if p.exists():
            return str(p.resolve())

        return cleaned

    return None
