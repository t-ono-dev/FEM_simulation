from __future__ import annotations

import socket
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter

from database import doi_to_url, normalize_doi
from journal_abbr import infer_journal_abbr


def fetch_bibtex_by_doi(doi: str, timeout: float = 10.0) -> str | None:
    """
    DOIからBibTeXを取得する。
    doi.orgのContent Negotiationを使うため、外部ネットワーク接続が必要。
    """
    doi = normalize_doi(doi)
    if not doi:
        return None

    url = doi_to_url(doi)
    if not url:
        return None

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/x-bibtex; charset=utf-8",
            "User-Agent": "LocalPaperManager/0.5 (mailto:local@example.com)",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout):
        return None

    for encoding in ("utf-8", "latin-1"):
        try:
            text = data.decode(encoding).strip()
            break
        except UnicodeDecodeError:
            continue
    else:
        return None

    if not text.startswith("@"):
        return None

    return text


def bibtex_text_to_record(bibtex_text: str) -> dict[str, Any] | None:
    try:
        db = bibtexparser.loads(bibtex_text)
    except Exception:
        return None

    if not db.entries:
        return None

    entry = db.entries[0]
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

    publication = clean_bibtex_text(publication)
    journal_abbr = clean_bibtex_text(journal_abbr)

    if publication and not journal_abbr:
        journal_abbr = infer_journal_abbr(publication)

    doi = normalize_doi(entry.get("doi"))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "title": title,
        "creators": authors,
        "year": year,
        "date": date,
        "publication": publication,
        "journal_abbr": journal_abbr,
        "doi": doi,
        "url": clean_bibtex_text(entry.get("url")) or doi_to_url(doi),
        "date_added": now,
        "pdf_path": None,
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


def fetch_bibtex_record_by_doi(doi: str, timeout: float = 10.0) -> dict[str, Any] | None:
    bibtex_text = fetch_bibtex_by_doi(doi, timeout=timeout)
    if not bibtex_text:
        return None
    return bibtex_text_to_record(bibtex_text)


def merge_pdf_record_with_bibtex(pdf_record: dict[str, Any], bibtex_record: dict[str, Any] | None) -> dict[str, Any]:
    """
    PDF取り込みで得たpdf_path, file_hash, date_addedは保持しつつ、
    BibTeXから得られた書誌情報で空欄を補完する。

    既存値がある項目は基本的に保持するが、PDFメタデータ由来のtitleより
    BibTeX titleの方が信頼できるため、titleはBibTeX優先。
    """
    if not bibtex_record:
        return pdf_record

    merged = dict(pdf_record)

    # PDF添付情報は必ず保持
    keep_from_pdf = {
        "pdf_path": pdf_record.get("pdf_path"),
        "file_hash": pdf_record.get("file_hash"),
        "date_added": pdf_record.get("date_added"),
    }

    for key, value in bibtex_record.items():
        if value is None or value == "":
            continue

        if key in {"pdf_path", "file_hash", "date_added"}:
            continue

        # titleはBibTeX優先
        if key == "title":
            merged[key] = value
            continue

        # その他は空欄のみ補完
        if not merged.get(key):
            merged[key] = value

    merged.update(keep_from_pdf)

    # DOIがある場合はURLを保証
    doi = normalize_doi(merged.get("doi"))
    if doi:
        merged["doi"] = doi
        if not merged.get("url"):
            merged["url"] = doi_to_url(doi)

    return merged


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
