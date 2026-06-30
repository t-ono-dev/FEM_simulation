from __future__ import annotations

import re


JOURNAL_ABBR_MAP = {
    "journal of applied physics": "J. Appl. Phys.",
    "applied physics letters": "Appl. Phys. Lett.",
    "applied physics express": "Appl. Phys. Express",
    "journal of physics d applied physics": "J. Phys. D: Appl. Phys.",
    "physical review": "Phys. Rev.",
    "physical review a": "Phys. Rev. A",
    "physical review b": "Phys. Rev. B",
    "physical review c": "Phys. Rev. C",
    "physical review d": "Phys. Rev. D",
    "physical review e": "Phys. Rev. E",
    "physical review letters": "Phys. Rev. Lett.",
    "physical review applied": "Phys. Rev. Appl.",
    "physical review materials": "Phys. Rev. Mater.",
    "physical review research": "Phys. Rev. Res.",
    "review of scientific instruments": "Rev. Sci. Instrum.",
    "reviews of modern physics": "Rev. Mod. Phys.",
    "journal of chemical physics": "J. Chem. Phys.",
    "journal of vacuum science and technology a": "J. Vac. Sci. Technol. A",
    "journal of vacuum science and technology b": "J. Vac. Sci. Technol. B",
    "surface science": "Surf. Sci.",
    "surface and interface analysis": "Surf. Interface Anal.",
    "semiconductor science and technology": "Semicond. Sci. Technol.",
    "solid state electronics": "Solid-State Electron.",
    "solid state communications": "Solid State Commun.",
    "physica status solidi a": "Phys. Status Solidi A",
    "physica status solidi b": "Phys. Status Solidi B",
    "physica status solidi rapid research letters": "Phys. Status Solidi RRL",
    "thin solid films": "Thin Solid Films",
    "materials science forum": "Mater. Sci. Forum",
    "ieee transactions on electron devices": "IEEE Trans. Electron Devices",
    "ieee electron device letters": "IEEE Electron Device Lett.",
    "japanese journal of applied physics": "Jpn. J. Appl. Phys.",
    "nanotechnology": "Nanotechnology",
    "nano letters": "Nano Lett.",
    "acs nano": "ACS Nano",
    "nature": "Nature",
    "nature materials": "Nat. Mater.",
    "nature nanotechnology": "Nat. Nanotechnol.",
    "nature electronics": "Nat. Electron.",
    "science": "Science",
    "science advances": "Sci. Adv.",
    "scientific reports": "Sci. Rep.",
    "carbon": "Carbon",
    "diamond and related materials": "Diam. Relat. Mater.",
    "applied surface science": "Appl. Surf. Sci.",
    "microelectronic engineering": "Microelectron. Eng.",
    "materials science in semiconductor processing": "Mater. Sci. Semicond. Process.",
}


WORD_ABBR = {
    "journal": "J.",
    "applied": "Appl.",
    "physics": "Phys.",
    "physical": "Phys.",
    "letters": "Lett.",
    "review": "Rev.",
    "reviews": "Rev.",
    "materials": "Mater.",
    "material": "Mater.",
    "science": "Sci.",
    "scientific": "Sci.",
    "technology": "Technol.",
    "technologies": "Technol.",
    "semiconductor": "Semicond.",
    "semiconductors": "Semicond.",
    "surface": "Surf.",
    "interface": "Interface",
    "interfaces": "Interfaces",
    "analysis": "Anal.",
    "solid": "Solid",
    "state": "State",
    "communications": "Commun.",
    "communication": "Commun.",
    "chemical": "Chem.",
    "chemistry": "Chem.",
    "vacuum": "Vac.",
    "instruments": "Instrum.",
    "instrumentation": "Instrum.",
    "electronics": "Electron.",
    "electronic": "Electron.",
    "electron": "Electron",
    "devices": "Devices",
    "device": "Device",
    "transactions": "Trans.",
    "japanese": "Jpn.",
    "nano": "Nano",
    "nanotechnology": "Nanotechnology",
    "carbon": "Carbon",
    "diamond": "Diam.",
    "related": "Relat.",
    "thin": "Thin",
    "films": "Films",
    "microelectronic": "Microelectron.",
    "engineering": "Eng.",
    "processing": "Process.",
    "status": "Status",
    "solidi": "Solidi",
    "rapid": "Rapid",
    "research": "Res.",
    "reports": "Rep.",
    "advances": "Adv.",
}


def normalize_journal_name(name: str | None) -> str:
    if not name:
        return ""
    text = name.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())
    return text


def infer_journal_abbr(publication: str | None) -> str | None:
    key = normalize_journal_name(publication)
    if not key:
        return None

    if key in JOURNAL_ABBR_MAP:
        return JOURNAL_ABBR_MAP[key]

    # よくある表記揺れ
    if key.startswith("journal of applied physics"):
        return "J. Appl. Phys."
    if key.startswith("applied physics letters"):
        return "Appl. Phys. Lett."
    if key.startswith("physical review b"):
        return "Phys. Rev. B"
    if key.startswith("physical review letters"):
        return "Phys. Rev. Lett."
    if key.startswith("physical review applied"):
        return "Phys. Rev. Appl."

    words = key.split()
    if not words:
        return None

    stop = {"of", "the", "and", "for", "in", "on", "at", "by"}
    abbr_words: list[str] = []

    for w in words:
        if w in stop:
            continue
        abbr_words.append(WORD_ABBR.get(w, w.capitalize()))

    result = " ".join(abbr_words).strip()
    return result or None
