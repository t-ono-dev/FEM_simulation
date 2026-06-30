from __future__ import annotations

import html
import re
from typing import Mapping, Any


def get_value(row: Mapping[str, Any] | Any, key: str) -> Any:
    try:
        if hasattr(row, "keys") and key in row.keys():
            return row[key]
    except Exception:
        pass

    try:
        return row.get(key)
    except Exception:
        return None


def format_aps_reference(row: Mapping[str, Any] | Any, number: int | None = None) -> str:
    """
    Plain-text fallback.
    Example:
    H. Yoshioka, T. Nakamura, and T. Kimoto, “Title”, J. Appl. Phys. 115, 014502 (2014).
    """
    return _format_aps_reference_core(row, number=number, html_mode=False)


def format_aps_reference_html(row: Mapping[str, Any] | Any, number: int | None = None) -> str:
    """
    Rich-text HTML reference.
    Volume is bold:
    J. Appl. Phys. <b>115</b>, 014502 (2014).
    """
    return _format_aps_reference_core(row, number=number, html_mode=True)


def _format_aps_reference_core(row: Mapping[str, Any] | Any, number: int | None = None, html_mode: bool = False) -> str:
    authors = format_authors(get_value(row, "creators") or "")
    title = clean(get_value(row, "title"))
    journal = clean(get_value(row, "journal_abbr")) or clean(get_value(row, "publication"))
    volume = clean(get_value(row, "volume"))
    pages = clean(get_value(row, "pages"))
    year = clean(get_value(row, "year")) or extract_year(clean(get_value(row, "date")))

    esc = html.escape if html_mode else (lambda x: x)

    prefix = f"[{number}] " if number is not None else ""
    parts: list[str] = []

    if authors:
        parts.append(esc(authors))

    if title:
        # APS風にタイトルを引用符で囲む
        parts.append(f"“{esc(title)}”")

    journal_part = ""
    if journal:
        journal_part = esc(journal)
        if volume:
            if html_mode:
                journal_part += f" <b>{esc(volume)}</b>"
            else:
                journal_part += f" {volume}"
        if pages:
            journal_part += f", {esc(pages)}"
        if year:
            journal_part += f" ({esc(year)})"
    elif year:
        journal_part = f"({esc(year)})"

    if journal_part:
        parts.append(journal_part)

    body = ", ".join(parts).strip()
    if not body:
        body = esc(title) if title else "Untitled"

    if not body.endswith("."):
        body += "."

    return esc(prefix) + body


def format_aps_references(rows: list[Mapping[str, Any] | Any]) -> str:
    return "\n".join(format_aps_reference(row, number=None) for row in rows)


def format_aps_references_html(rows: list[Mapping[str, Any] | Any]) -> str:
    body = "<br>\n".join(format_aps_reference_html(row, number=None) for row in rows)
    return f"<html><body>{body}</body></html>"


def clean(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\&", "&")
    text = text.replace("--", "-")
    text = " ".join(text.split())
    return text or None


def extract_year(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return match.group(0)
    return None


def format_authors(author_text: str) -> str:
    author_text = clean(author_text) or ""
    if not author_text:
        return ""

    if " and " in author_text:
        names = [n.strip() for n in author_text.split(" and ") if n.strip()]
    else:
        names = [n.strip() for n in re.split(r";\s*", author_text) if n.strip()]

    if not names:
        return ""

    if len(names) > 10:
        return format_single_author(names[0]) + " et al."

    formatted = [format_single_author(n) for n in names]

    if len(formatted) == 1:
        return formatted[0]

    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"

    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def format_single_author(name: str) -> str:
    name = clean(name) or ""
    if not name:
        return ""

    if re.match(r"^[A-Z]\.", name):
        return name

    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        return f"{initials(first)} {last}".strip()

    words = name.split()
    if len(words) == 1:
        return words[0]

    last = words[-1]
    first_middle = " ".join(words[:-1])
    return f"{initials(first_middle)} {last}".strip()


def initials(first_middle: str) -> str:
    tokens = [t for t in re.split(r"[\s\-]+", first_middle.strip()) if t]
    out: list[str] = []
    lowercase_particles = {"van", "von", "de", "del", "da", "di", "la", "le"}

    for token in tokens:
        token = token.strip(".")
        if not token:
            continue
        if token.lower() in lowercase_particles:
            continue
        if len(token) == 1:
            out.append(token.upper() + ".")
        else:
            out.append(token[0].upper() + ".")

    return " ".join(out)
