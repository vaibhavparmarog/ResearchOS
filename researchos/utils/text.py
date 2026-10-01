"""Text cleaning and matching helpers shared across the pipeline."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "", "­": "",
}
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


_DETACHED_ACCENT = re.compile(r"(?<=\w) ?([̀-ͯ])([A-Za-z])")


def normalize_chars(text: str) -> str:
    """Unify unicode quirks that PDFs commonly introduce (ligatures, smart quotes, NBSP, detached accents)."""
    text = unicodedata.normalize("NFKC", text)
    # LaTeX PDFs often store "é" as a spacing accent before the letter ("D ́epartement"): re-attach it.
    text = _DETACHED_ACCENT.sub(lambda m: unicodedata.normalize("NFC", m.group(2) + m.group(1)), text)
    for src, dst in {**_QUOTES, **_LIGATURES}.items():
        text = text.replace(src, dst)
    # drop control characters except newline / tab
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def clean_page_text(text: str) -> str:
    """Repair line-break hyphenation and whitespace without changing the wording."""
    text = normalize_chars(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # "informa-\ntion" -> "information"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_repeated_lines(pages: list[str], min_pages: int = 4, share: float = 0.5, edge: int = 2) -> list[str]:
    """Remove running headers/footers: short lines repeated at the top/bottom of many pages.

    Only the first/last `edge` non-empty lines of a page are candidates, so repeated wording in
    the body is never touched. Page numbers are normalised (digits -> '#') so "Page 3" and
    "Page 4" count as the same line.
    """
    if len(pages) < min_pages:
        return pages

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    def edge_indexes(lines: list[str]) -> set[int]:
        idx = [i for i, l in enumerate(lines) if l.strip()]
        return set(idx[:edge] + idx[-edge:])

    counts: Counter[str] = Counter()
    for text in pages:
        lines = text.split("\n")
        counts.update({key(lines[i]) for i in edge_indexes(lines) if len(lines[i].strip()) <= 80})
    threshold = max(3, int(len(pages) * share))
    repeated = {k for k, c in counts.items() if c >= threshold}
    if not repeated:
        return pages
    out = []
    for text in pages:
        lines = text.split("\n")
        drop = {i for i in edge_indexes(lines) if len(lines[i].strip()) <= 80 and key(lines[i]) in repeated}
        out.append("\n".join(l for i, l in enumerate(lines) if i not in drop).strip())
    return out


def match_form(text: str) -> str:
    """Aggressive normalisation used ONLY for verifying that a quote occurs in a page."""
    text = normalize_chars(text).casefold()
    text = re.sub(r"[^\w%.$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def format_pages(pages: list[int]) -> str:
    """[4,5,6,9] -> 'Pages 4-6, 9'."""
    if not pages:
        return ""
    pages = sorted(set(pages))
    runs, start, prev = [], pages[0], pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        runs.append((start, prev))
        start = prev = p
    runs.append((start, prev))
    parts = [str(a) if a == b else f"{a}–{b}" for a, b in runs]
    return ("Page " if len(pages) == 1 else "Pages ") + ", ".join(parts)
