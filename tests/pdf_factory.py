"""Builds throw-away PDFs for tests. Nothing here is shipped to, or read by, the application."""

from __future__ import annotations

import pymupdf as fitz


def make_pdf(pages: list[str], title: str | None = None, header: str | None = None) -> bytes:
    """One text page per list item ("" produces a blank page). Optional running header."""
    doc = fitz.open()
    for i, text in enumerate(pages, 1):
        page = doc.new_page()
        if header:
            page.insert_text((72, 40), f"{header} - page {i}", fontsize=8)
        if text:
            rc = page.insert_textbox(fitz.Rect(72, 72, 523, 770), text, fontsize=11)
            assert rc >= 0, "test page text overflowed the page"
    if title:
        doc.set_metadata({"title": title})
    data = doc.tobytes()
    doc.close()
    return data


def make_image_only_pdf() -> bytes:
    """A PDF whose only content is a drawn shape (no text layer), like a scan."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(100, 100, 300, 300), fill=(0.2, 0.4, 0.8))
    data = doc.tobytes()
    doc.close()
    return data
