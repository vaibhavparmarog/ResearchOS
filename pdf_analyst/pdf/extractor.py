"""PDF ingestion: validation, page-by-page text extraction, light structure detection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from statistics import median

import pymupdf as fitz  # PyMuPDF

from ..utils.config import Limits
from ..utils.errors import (
    CorruptPDFError,
    EmptyPDFError,
    EncryptedPDFError,
    FileTooLargeError,
    InvalidFileTypeError,
    NoFileError,
    TooManyPagesError,
)
from ..utils.text import clean_page_text, strip_repeated_lines

MIN_CHARS_FOR_TEXT_PAGE = 25   # pages with fewer characters are treated as empty / image-only


@dataclass
class Page:
    number: int          # 1-based, as printed by PDF viewers
    text: str
    has_text: bool


@dataclass
class Heading:
    text: str
    level: int
    page: int


@dataclass
class ExtractedDocument:
    filename: str
    size_bytes: int
    sha256: str
    page_count: int
    pages: list[Page]
    metadata: dict = field(default_factory=dict)      # title/author/... as declared inside the PDF
    headings: list[Heading] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text_pages(self) -> list[Page]:
        return [p for p in self.pages if p.has_text]

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def page_text(self, number: int) -> str:
        return self.pages[number - 1].text if 1 <= number <= self.page_count else ""


def _detect_headings(doc: "fitz.Document", max_pages: int = 400) -> list[Heading]:
    """Headings from the PDF outline if present, else from font-size analysis."""
    toc = doc.get_toc(simple=True)
    if toc:
        return [Heading(t.strip(), lvl, pg) for lvl, t, pg in toc if t.strip() and 1 <= pg <= doc.page_count][:200]

    # Font-size heuristic: lines set noticeably larger than the median body size.
    lines: list[tuple[float, str, int, bool]] = []  # (size, text, page, bold)
    for pno in range(min(doc.page_count, max_pages)):
        try:
            data = doc[pno].get_text("dict")
        except Exception:
            continue
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans)
                size = max(s["size"] for s in spans)
                bold = all(bool(s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                lines.append((size, text, pno + 1, bold))
    if not lines:
        return []
    body = median(s for s, *_ in lines)
    sizes = sorted({round(s) for s, *_ in lines if s >= body * 1.15}, reverse=True)
    headings: list[Heading] = []
    for size, text, page, bold in lines:
        if not (3 <= len(text) <= 120) or re.fullmatch(r"[\d\W]+", text):
            continue
        if size >= body * 1.15:
            headings.append(Heading(text, 1 + sizes.index(round(size)) if round(size) in sizes else 1, page))
        elif bold and size >= body * 0.98 and len(text) <= 70 and not text.endswith((".", ",")):
            headings.append(Heading(text, len(sizes) + 1, page))
    return headings[:200]


def extract_pdf(
    data: bytes | None,
    filename: str = "uploaded.pdf",
    limits: Limits | None = None,
) -> ExtractedDocument:
    """Validate and parse a PDF supplied at runtime. Raises PDFError subclasses."""
    limits = limits or Limits.from_env()

    if not data:
        raise NoFileError()
    if not filename.lower().endswith(".pdf"):
        raise InvalidFileTypeError()
    if len(data) > limits.max_pdf_mb * 1024 * 1024:
        raise FileTooLargeError(
            f"The file is {len(data) / 1024 / 1024:.1f} MB; the limit is {limits.max_pdf_mb} MB."
        )
    if not data.lstrip()[:1024].startswith(b"%PDF-") and b"%PDF-" not in data[:1024]:
        raise InvalidFileTypeError(
            "The file has a .pdf extension but its contents are not a PDF document."
        )

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # fitz raises several private exception types
        raise CorruptPDFError(detail=repr(exc)) from exc

    try:
        if doc.needs_pass:
            raise EncryptedPDFError()
        if doc.page_count == 0:
            raise EmptyPDFError("The PDF has no pages.")
        if doc.page_count > limits.max_pages:
            raise TooManyPagesError(
                f"The PDF has {doc.page_count} pages; the limit is {limits.max_pages}. "
                "Split it into smaller parts and upload each part."
            )

        raw_pages: list[str] = []
        warnings: list[str] = []
        broken = 0
        for pno in range(doc.page_count):
            try:
                raw_pages.append(doc[pno].get_text("text", sort=True))
            except Exception:
                raw_pages.append("")
                broken += 1
        if broken == doc.page_count:
            raise CorruptPDFError("The PDF's pages could not be read; the file is likely damaged.")
        if broken:
            warnings.append(f"{broken} page(s) could not be read and were skipped.")

        cleaned = strip_repeated_lines([clean_page_text(t) for t in raw_pages])
        pages = [
            Page(number=i + 1, text=t, has_text=len(t) >= MIN_CHARS_FOR_TEXT_PAGE)
            for i, t in enumerate(cleaned)
        ]
        n_text = sum(p.has_text for p in pages)
        if n_text == 0:
            raise EmptyPDFError(
                "No text could be extracted. This PDF looks like a scan or image-only document, "
                "and OCR is not supported. Upload a PDF with selectable text."
            )
        if n_text < len(pages):
            missing = [p.number for p in pages if not p.has_text]
            shown = ", ".join(map(str, missing[:15])) + ("..." if len(missing) > 15 else "")
            warnings.append(
                f"{len(missing)} page(s) had no extractable text (blank or image-only) and are "
                f"not covered by the analysis: {shown}."
            )

        meta = {k: v.strip() for k, v in (doc.metadata or {}).items() if isinstance(v, str) and v.strip()}
        try:
            headings = _detect_headings(doc)
        except Exception:
            headings = []

        return ExtractedDocument(
            filename=filename,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            page_count=doc.page_count,
            pages=pages,
            metadata=meta,
            headings=headings,
            warnings=warnings,
        )
    finally:
        doc.close()
