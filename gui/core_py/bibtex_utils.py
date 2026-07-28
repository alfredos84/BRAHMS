"""
bibtex_utils
============
Best-effort conversion of a single BibTeX entry into a short citation
string, e.g. "Gayer et al., Appl. Phys. B 91, 343 (2008)", matching the
style used for the preloaded crystals in crystal_db.py.
"""

import re

# Common optics/physics journal abbreviations. Unlisted journals are used
# verbatim (as typed in the BibTeX "journal" field).
_JOURNAL_ABBREV = {
    "applied physics b": "Appl. Phys. B",
    "applied physics letters": "Appl. Phys. Lett.",
    "journal of applied physics": "J. Appl. Phys.",
    "optics letters": "Opt. Lett.",
    "optics express": "Opt. Express",
    "optics communications": "Opt. Commun.",
    "physical review a": "Phys. Rev. A",
    "physical review b": "Phys. Rev. B",
    "physical review letters": "Phys. Rev. Lett.",
    "journal of the optical society of america b": "J. Opt. Soc. Am. B",
    "journal of the optical society of america": "J. Opt. Soc. Am.",
    "ieee journal of quantum electronics": "IEEE J. Quantum Electron.",
    "nature photonics": "Nat. Photonics",
    "nature": "Nature",
    "science": "Science",
}

_FIELD_RE = re.compile(
    r'(\w+)\s*=\s*[{"]((?:[^{}"]|\{[^{}]*\})*)[}"]', re.IGNORECASE)


def _first_page(pages: str) -> str:
    if not pages:
        return ""
    return re.split(r"--|-|,", pages.strip())[0].strip()


def _first_author_surname(author_field: str) -> tuple[str, bool]:
    """Return (surname, has_multiple_authors)."""
    if not author_field:
        return "", False
    authors = re.split(r"\s+and\s+", author_field.strip(), flags=re.IGNORECASE)
    first = authors[0].strip()
    if "," in first:
        surname = first.split(",")[0].strip()
    else:
        parts = first.split()
        surname = parts[-1] if parts else first
    return surname, len(authors) > 1


def parse_bibtex_fields(bibtex_str: str) -> dict:
    """
    Best-effort extraction of key = {value} / key = "value" pairs from a
    single BibTeX entry. Returns a dict with lowercase keys; unmatched
    fields are simply absent.
    """
    fields = {}
    for m in _FIELD_RE.finditer(bibtex_str):
        key = m.group(1).strip().lower()
        val = re.sub(r"\s+", " ", m.group(2).strip())
        fields[key] = val
    return fields


def bibtex_to_citation(bibtex_str: str) -> str:
    """
    Convert a BibTeX entry into a short citation string, e.g.
    "Gayer et al., Appl. Phys. B 91, 343 (2008)".
    Returns "" if the input is empty or has no parseable year.
    """
    bibtex_str = (bibtex_str or "").strip()
    if not bibtex_str:
        return ""
    fields = parse_bibtex_fields(bibtex_str)
    year = fields.get("year", "").strip()
    if not year:
        return ""

    parts = []
    surname, multiple = _first_author_surname(fields.get("author", ""))
    if surname:
        parts.append(f"{surname} et al." if multiple else surname)

    journal = fields.get("journal", "").strip()
    journal_abbrev = _JOURNAL_ABBREV.get(journal.lower(), journal)
    volume = fields.get("volume", "").strip()
    page = _first_page(fields.get("pages", ""))

    if journal_abbrev:
        vol_page = journal_abbrev
        if volume:
            vol_page += f" {volume}"
        if page:
            vol_page += f", {page}"
        parts.append(vol_page)

    if not parts:
        return f"({year})"
    return ", ".join(parts) + f" ({year})"
