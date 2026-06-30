from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader


DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def safe_filename(name: str) -> str:
    forbidden = '<>:"/\\|?*'
    for ch in forbidden:
        name = name.replace(ch, "_")
    return name.strip() or "untitled"


def extract_pdf_metadata(path: Path) -> dict[str, str | None]:
    title = None
    creators = None
    doi = None

    try:
        reader = PdfReader(str(path))
        meta = reader.metadata

        if meta:
            raw_title = getattr(meta, "title", None)
            raw_author = getattr(meta, "author", None)
            title = clean_text(raw_title) if raw_title else None
            creators = clean_text(raw_author) if raw_author else None

        text_parts: list[str] = []
        for page in reader.pages[:2]:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue

        joined = "\n".join(text_parts)
        match = DOI_PATTERN.search(joined)
        if match:
            doi = match.group(0).rstrip(".,;)")
    except Exception:
        pass

    return {
        "title": title,
        "creators": creators,
        "doi": doi,
    }


def import_pdf_file(source_path: Path, library_dir: Path) -> dict[str, str | None]:
    source_path = source_path.resolve()
    library_dir.mkdir(parents=True, exist_ok=True)

    file_hash = sha256_file(source_path)
    metadata = extract_pdf_metadata(source_path)

    title_for_filename = metadata.get("title") or source_path.stem
    filename = safe_filename(title_for_filename)[:120] + ".pdf"
    dest_path = library_dir / filename

    if dest_path.exists():
        dest_path = library_dir / f"{safe_filename(title_for_filename)[:100]}_{file_hash[:8]}.pdf"

    if source_path != dest_path.resolve():
        shutil.copy2(source_path, dest_path)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "title": metadata.get("title") or source_path.stem,
        "creators": metadata.get("creators"),
        "year": None,
        "date": None,
        "publication": None,
        "journal_abbr": None,
        "doi": metadata.get("doi"),
        "url": None,
        "date_added": now,
        "pdf_path": str(dest_path.resolve()),
        "bibtex_key": None,
        "bibtex_raw": None,
        "file_hash": file_hash,
        "notes": None,
        "volume": None,
        "pages": None,
        "issue": None,
        "publisher": None,
        "entry_type": "article",
    }


def clean_text(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())
