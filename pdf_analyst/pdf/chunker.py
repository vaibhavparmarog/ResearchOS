"""Page-aware chunking. Every chunk keeps explicit [[Page N]] markers so the model can cite pages."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.config import Limits
from .extractor import ExtractedDocument


def page_marker(n: int) -> str:
    return f"[[Page {n}]]"


@dataclass
class Chunk:
    index: int
    pages: list[int]          # pages that contribute text to this chunk
    text: str                 # text with [[Page N]] markers

    @property
    def first_page(self) -> int:
        return self.pages[0]

    @property
    def last_page(self) -> int:
        return self.pages[-1]

    @property
    def label(self) -> str:
        return f"pages {self.first_page}-{self.last_page}" if self.first_page != self.last_page else f"page {self.first_page}"


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split one oversized page on paragraph, then sentence, then hard boundaries."""
    if len(text) <= max_chars:
        return [text]
    parts, buf = [], ""
    for para in re.split(r"\n{2,}|\n(?=[A-Z0-9])", text):
        if len(para) > max_chars:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                while len(sent) > max_chars:
                    parts.append(sent[:max_chars])
                    sent = sent[max_chars:]
                if len(buf) + len(sent) + 1 > max_chars and buf:
                    parts.append(buf)
                    buf = ""
                buf = f"{buf} {sent}".strip()
            continue
        if len(buf) + len(para) + 2 > max_chars and buf:
            parts.append(buf)
            buf = ""
        buf = f"{buf}\n\n{para}".strip()
    if buf:
        parts.append(buf)
    return parts


def chunk_document(doc: ExtractedDocument, limits: Limits | None = None) -> list[Chunk]:
    """Greedily pack whole pages into chunks of ~chunk_chars; oversized pages are split."""
    limits = limits or Limits.from_env()
    max_chars = limits.chunk_chars
    chunks: list[Chunk] = []
    cur_pages: list[int] = []
    cur_parts: list[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur_pages, cur_parts, cur_len
        if cur_parts:
            chunks.append(Chunk(len(chunks), cur_pages, "\n\n".join(cur_parts)))
        cur_pages, cur_parts, cur_len = [], [], 0

    for page in doc.text_pages:
        for piece in _split_long(page.text, max_chars):
            block = f"{page_marker(page.number)}\n{piece}"
            if cur_parts and cur_len + len(block) > max_chars:
                flush()
            cur_parts.append(block)
            if page.number not in cur_pages:
                cur_pages.append(page.number)
            cur_len += len(block) + 2
    flush()
    return chunks


def full_text(doc: ExtractedDocument) -> str:
    """Whole document with page markers (used for single-pass analysis of short documents)."""
    return "\n\n".join(f"{page_marker(p.number)}\n{p.text}" for p in doc.text_pages)
