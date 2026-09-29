"""Report rendering: Markdown (also used by the UI), plain text, JSON, and PDF."""

from __future__ import annotations

import io
import json
import re

import markdown as _markdown

from ..utils.text import format_pages
from .models import Report

BASIS_LABEL = {
    "document": "Document content",
    "interpretation": "AI interpretation (not stated in the document)",
    "mixed": "Document content with AI interpretation",
}


def _human_size(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


def overview_rows(r: Report) -> list[tuple[str, str]]:
    o = r.overview
    title_note = {"document": "", "pdf_metadata": " (from PDF metadata)", "filename": " (from filename; no title found in text)"}[o.title_source]
    return [
        ("Title", o.title + title_note),
        ("Author(s)", ", ".join(o.authors) if o.authors else "Not stated in the document."),
        ("Date / year", o.date or "Not stated in the document."),
        ("Document type", o.document_type),
        ("Pages", str(o.page_count)),
        ("File", f"{o.filename} ({_human_size(o.size_bytes)})"),
    ]


def section_meta_line(s) -> str:
    parts = [f"*Basis: {BASIS_LABEL[s.basis]}*"]
    if s.pages:
        parts.append(f"*Source: {format_pages(s.pages)}*")
    if not s.supported:
        parts.append("*⚠ No verifiable page reference — treat with caution*")
    return " · ".join(parts)


def to_markdown(r: Report) -> str:
    out = [f"# {r.overview.title}", "", "## Document Overview", "", "| Field | Value |", "|---|---|"]
    out += [f"| {k} | {v.replace('|', '/')} |" for k, v in overview_rows(r)]
    for s in r.sections:
        out += ["", f"## {s.heading}", "", section_meta_line(s), "", s.body_markdown.strip()]
        if s.evidence:
            out += ["", "**Supporting quotes (verified verbatim against the PDF):**", ""]
            out += [f"> “{e.quote}” — {format_pages([e.page])}" for e in s.evidence]
    if r.metrics:
        out += ["", "## Important Data / Metrics", "", "| Metric | Value | Source | Verified |", "|---|---|---|---|"]
        for m in r.metrics:
            source = format_pages(m.pages) or "—"
            status = "yes" if m.verified else "NOT FOUND in PDF"
            out.append(f"| {m.label.replace('|', '/')} | {m.value.replace('|', '/')} | {source} | {status} |")
    if r.not_stated:
        out += ["", "## Not Stated in the Document", ""]
        out += [f"- {x}: Not stated in the document." for x in r.not_stated]
    out += ["", "## Source References", ""]
    if r.references:
        for ref in r.references:
            out.append(f"- **{format_pages([ref.page])}** — used in: {', '.join(ref.used_in)}")
            out += [f"    - “{q}”" for q in ref.quotes]
    else:
        out.append("No page references could be verified.")
    out += ["", "## Verification Notes", ""]
    out += [f"- {n}" for n in r.validation_notes] or ["- All page references and quotes were verified against the PDF; nothing needed correcting."]
    for w in r.warnings:
        out.append(f"- Coverage note: {w}")
    m = r.meta
    out += [
        "",
        "---",
        f"*Generated {m.get('generated_at', '')} by {m.get('model', 'an AI model')} "
        f"({m.get('strategy', '')}, {m.get('chunks', 0)} chunk(s), {m.get('llm_calls', 0)} LLM call(s)) "
        f"from the uploaded file {r.overview.filename}. Sections marked as AI interpretation are not statements by the document.*",
    ]
    return "\n".join(out) + "\n"


def to_text(r: Report) -> str:
    """Plain text: same content as Markdown with the markup removed."""
    md = to_markdown(r)
    md = re.sub(r"^#{1,6}\s*", "", md, flags=re.M)
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    md = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", md)
    md = re.sub(r"^\|[-| :]+\|\s*$\n?", "", md, flags=re.M)
    md = re.sub(r"^\|\s*|\s*\|\s*$", "", md, flags=re.M).replace(" | ", "  |  ")
    md = re.sub(r"^> ", "    ", md, flags=re.M)
    return re.sub(r"^---$", "-" * 60, md, flags=re.M)


def to_json(r: Report) -> str:
    return json.dumps(r.to_dict(), indent=2, ensure_ascii=False)


_PDF_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.4; }
h1 { font-size: 20pt; } h2 { font-size: 14pt; margin-top: 14pt; border-bottom: 1px solid #999; }
table { border-collapse: collapse; } td, th { border: 1px solid #999; padding: 3pt 6pt; }
blockquote { color: #444; margin-left: 12pt; }
"""


def to_pdf(r: Report) -> bytes:
    """Render the report to PDF with PyMuPDF's HTML layout engine."""
    import pymupdf as fitz

    html = _markdown.markdown(to_markdown(r), extensions=["tables", "sane_lists"])
    story = fitz.Story(html=html, user_css=_PDF_CSS)
    stream = io.BytesIO()
    writer = fitz.DocumentWriter(stream)
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (48, 48, -48, -48)
    more = 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return stream.getvalue()
