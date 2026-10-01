"""Page text in reading order, plus text-quality scoring and optional OCR.

Research papers are mostly two-column. PyMuPDF's `sort=True` sorts lines by position, which
interleaves the two columns line by line, so we read blocks in content-stream order (correct for
LaTeX/Word output) and fall back to explicit column ordering only when the stream order looks
disordered. Rotated lines (arXiv margin stamps, axis labels) are dropped.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import pymupdf as fitz

log = logging.getLogger(__name__)

TEXT_FLAGS = fitz.TEXTFLAGS_TEXT & ~fitz.TEXT_PRESERVE_IMAGES


@dataclass
class Block:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def page_blocks(page: "fitz.Page", textpage=None) -> list[Block]:
    """Text blocks in content-stream order, without rotated lines."""
    data = page.get_text("dict", flags=TEXT_FLAGS, textpage=textpage)
    blocks: list[Block] = []
    for b in data.get("blocks", []):
        if b.get("type") != 0:
            continue
        lines = []
        for line in b.get("lines", []):
            dx, _dy = line.get("dir", (1, 0))
            if dx < 0.98:                       # vertical / rotated text
                continue
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if text:
                lines.append(text)
        if lines:
            x0, y0, x1, y1 = b["bbox"]
            blocks.append(Block(x0, y0, x1, y1, "\n".join(lines)))
    return blocks


def _column_of(b: Block, width: float) -> str:
    mid = width / 2
    if (b.x1 - b.x0) > 0.55 * width or (b.x0 < mid - 0.08 * width and b.x1 > mid + 0.08 * width):
        return "full"
    return "left" if (b.x0 + b.x1) / 2 < mid else "right"


def _is_two_column(blocks: list[Block], width: float) -> bool:
    total = sum(len(b.text) for b in blocks) or 1
    left = sum(len(b.text) for b in blocks if _column_of(b, width) == "left")
    right = sum(len(b.text) for b in blocks if _column_of(b, width) == "right")
    return left / total >= 0.15 and right / total >= 0.15


def stream_disorder(blocks: list[Block], width: float, height: float) -> float:
    """Share of consecutive same-column block pairs where the stream jumps back up the page."""
    pairs = jumps = 0
    for a, b in zip(blocks, blocks[1:]):
        if _column_of(a, width) != _column_of(b, width) or _column_of(a, width) == "full":
            continue
        pairs += 1
        if b.y0 < a.y0 - 0.15 * height:
            jumps += 1
    return jumps / pairs if pairs else 0.0


def column_order(blocks: list[Block], width: float) -> list[Block]:
    """Geometric reading order: per band between full-width blocks, left column then right."""
    if not _is_two_column(blocks, width):
        return sorted(blocks, key=lambda b: (round(b.y0), b.x0))
    full = sorted((b for b in blocks if _column_of(b, width) == "full"), key=lambda b: b.y0)
    cols = [b for b in blocks if _column_of(b, width) != "full"]
    out: list[Block] = []
    prev = float("-inf")
    for f in [*full, None]:
        limit = f.y0 if f else float("inf")
        band = [b for b in cols if prev <= b.y0 < limit]
        for side in ("left", "right"):
            out += sorted((b for b in band if _column_of(b, width) == side), key=lambda b: b.y0)
        if f:
            out.append(f)
            prev = f.y0
    return out


def page_text(page: "fitz.Page", textpage=None) -> str:
    blocks = page_blocks(page, textpage)
    w, h = page.rect.width, page.rect.height
    if blocks and stream_disorder(blocks, w, h) > 0.3:
        blocks = column_order(blocks, w)
    return "\n\n".join(b.text for b in blocks)


# ----------------------------------------------------------------------------- quality / OCR
_WORD = re.compile(r"[^\W\d_]{2,}")
_VOWELS = set("aeiouyAEIOUYàáâäèéêëìíîïòóôöùúûü")


def text_quality(text: str) -> float:
    """0..1: share of whitespace tokens that look like real words.

    Normal prose scores ~0.7-0.9, maths-heavy pages ~0.4-0.6, text from fonts without a
    Unicode mapping (e.g. "$&%('*) +-,/.1012") scores well under 0.3.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    good = 0
    for t in tokens:
        letters = sum(ch.isalpha() for ch in t)
        if letters >= 2 and letters >= 0.7 * len(t) and any(ch in _VOWELS for ch in t) and _WORD.search(t):
            good += 1
    return good / len(tokens)


def ocr_available() -> bool:
    if os.environ.get("OCR_MODE", "auto").lower() == "off":
        return False
    try:
        return bool(fitz.get_tessdata())
    except Exception:
        return False


def ocr_page_text(page: "fitz.Page", dpi: int | None = None) -> str:
    """OCR one page with Tesseract via PyMuPDF. Raises if OCR is unavailable."""
    dpi = dpi or int(os.environ.get("OCR_DPI", "150") or 150)
    tp = page.get_textpage_ocr(language=os.environ.get("OCR_LANGUAGE", "eng"), dpi=dpi, full=True)
    return page_text(page, textpage=tp)
