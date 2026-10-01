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
from ..utils.text import clean_page_text, normalize_chars, strip_repeated_lines
from .layout import ocr_available, ocr_page_text, page_text, text_quality

MIN_CHARS_FOR_TEXT_PAGE = 25   # pages with fewer characters are treated as empty / image-only
GARBLED_PAGE_QUALITY = 0.2     # below this a page's text layer is treated as unreadable
GARBLED_DOC_MEDIAN = 0.3       # below this median the whole text layer is treated as unreadable

_REFS_START = re.compile(
    r"(?m)^[ \t]*(?:\d{1,2}\.?|[IVX]{1,4}\.)?[ \t]*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Literature Cited|REFERENCES AND NOTES)[ \t]*$"
)
_REFS_END = re.compile(r"(?m)^[ \t]*(?:Appendix|APPENDIX|Appendices|APPENDICES|Supplementary Material|SUPPLEMENTARY MATERIAL)\b")


@dataclass
class Page:
    number: int          # 1-based, as printed by PDF viewers
    text: str            # full cleaned text: used for display and for verifying quotes
    has_text: bool
    analysis_text: str | None = None   # text sent to the LLM (references removed); None = same as text
    ocr: bool = False

    @property
    def for_analysis(self) -> str:
        return self.text if self.analysis_text is None else self.analysis_text


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
    reference_pages: list[int] = field(default_factory=list)   # pages holding the bibliography
    ocr_pages: list[int] = field(default_factory=list)

    @property
    def text_pages(self) -> list[Page]:
        return [p for p in self.pages if p.has_text]

    @property
    def analysis_pages(self) -> list[Page]:
        """Pages with text that goes to the LLM (bibliography-only pages are excluded)."""
        return [p for p in self.pages if p.has_text and p.for_analysis.strip()]

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
    return _font_headings(doc, 1, max_pages)


def _font_headings(doc: "fitz.Document", first_page: int = 1, max_pages: int = 400) -> list[Heading]:
    """Heuristic headings: lines set noticeably larger than body text, or short bold lines."""
    lines: list[tuple[float, str, int, bool]] = []  # (size, text, page, bold)
    for pno in range(first_page - 1, min(doc.page_count, first_page - 1 + max_pages)):
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


def _refs_end(page: Page, begin: int, start_page: int, headings: list[Heading]) -> int | None:
    """Offset in `page.text` where the bibliography ends on this page, if it does.

    Ends at an explicit "Appendix"/"Supplementary" line, or at the first font-detected heading
    after the references heading (reference entries are body text; appendix titles such as
    "A Visualisations" or "Attention Visualizations" are set as headings).
    """
    candidates = []
    m = _REFS_END.search(page.text, begin)
    if m:
        candidates.append(m.start())
    search_from = begin + 1 if page.number == start_page else 0
    for h in headings:
        text = h.text.strip()
        if h.page != page.number or _REFS_START.fullmatch(text) or len(text) < 4:
            continue
        pos = _find_heading(page.text, text, search_from)
        if pos is not None:
            candidates.append(pos)
    return min(candidates) if candidates else None


def _find_heading(text: str, heading: str, start: int) -> int | None:
    """Locate a heading in page text, tolerating line breaks and a missing number/letter prefix."""
    words = heading.split()
    variants = [words]
    if len(words) > 1 and re.fullmatch(r"[A-Z]|\d+|[A-Z0-9]+(?:\.\d+)+\.?", words[0]):
        variants.append(words[1:])           # "A Visualisations" -> "Visualisations"
    for v in variants:
        if not v or len(" ".join(v)) < 4:
            continue
        m = re.compile(r"(?m)^[ \t]*" + r"\s+".join(map(re.escape, v))).search(text, start)
        if m:
            return m.start()
    return None


def _mark_references(pages: list[Page], headings: list[Heading] | None = None) -> list[int]:
    """Exclude the bibliography from analysis text. Returns the pages it spans.

    The section starts at the last standalone "References"/"Bibliography" heading in the second
    half of the document and ends at the next appendix-like heading (see `_refs_end`) or the end.
    """
    headings = headings or []
    text_pages = [p for p in pages if p.has_text]
    if len(text_pages) < 3:
        return []
    start: tuple[int, int] | None = None
    half = text_pages[len(text_pages) // 2].number
    for p in text_pages:
        if p.number < half - 1:
            continue
        for m in _REFS_START.finditer(p.text):
            start = (p.number, m.start())
    if not start:
        return []
    spanned: list[int] = []
    for p in text_pages:
        if p.number < start[0]:
            continue
        begin = start[1] if p.number == start[0] else 0
        end_at = _refs_end(p, begin, start[0], headings)
        end = end_at if end_at is not None else len(p.text)
        p.analysis_text = (p.text[:begin] + p.text[end:]).strip()
        spanned.append(p.number)
        if end_at is not None:
            break
    return spanned


def extract_pdf(
    data: bytes | None,
    filename: str = "uploaded.pdf",
    limits: Limits | None = None,
    on_progress=None,
) -> ExtractedDocument:
    """Validate and parse a PDF supplied at runtime. Raises PDFError subclasses.

    `on_progress(fraction, message)` is called during OCR, the only slow step.
    """
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
                raw_pages.append(page_text(doc[pno]))
            except Exception:
                raw_pages.append("")
                broken += 1
        if broken == doc.page_count:
            raise CorruptPDFError("The PDF's pages could not be read; the file is likely damaged.")
        if broken:
            warnings.append(f"{broken} page(s) could not be read and were skipped.")

        # Pages whose text layer is missing or unreadable (scans, fonts without a Unicode map).
        quality = [text_quality(t) if len(t.strip()) >= MIN_CHARS_FOR_TEXT_PAGE else 0.0 for t in raw_pages]
        scored = sorted(q for t, q in zip(raw_pages, quality) if len(t.strip()) >= MIN_CHARS_FOR_TEXT_PAGE)
        doc_garbled = bool(scored) and scored[len(scored) // 2] < GARBLED_DOC_MEDIAN
        unreadable = [
            i for i, (t, q) in enumerate(zip(raw_pages, quality))
            if len(t.strip()) < MIN_CHARS_FOR_TEXT_PAGE or q < GARBLED_PAGE_QUALITY or doc_garbled
        ]
        ocr_done: list[int] = []
        if unreadable and ocr_available():
            todo = unreadable[: limits.max_ocr_pages]
            for n, i in enumerate(todo, 1):
                if on_progress:
                    on_progress(n / len(todo), f"Running OCR on page {i + 1} ({n} of {len(todo)})")
                try:
                    text = ocr_page_text(doc[i])
                except Exception:
                    continue
                if len(text.strip()) >= MIN_CHARS_FOR_TEXT_PAGE and text_quality(text) >= GARBLED_PAGE_QUALITY:
                    raw_pages[i] = text
                    ocr_done.append(i)
            if len(unreadable) > limits.max_ocr_pages:
                warnings.append(f"Only the first {limits.max_ocr_pages} unreadable pages were OCR'd (MAX_OCR_PAGES).")
        still_bad = set(unreadable) - set(ocr_done)
        if ocr_done:
            warnings.append(
                f"{len(ocr_done)} page(s) had no usable text layer and were read with OCR; "
                "expect occasional recognition errors on those pages."
            )

        cleaned = strip_repeated_lines([clean_page_text(t) for t in raw_pages])
        pages = [
            Page(number=i + 1, text=t, has_text=len(t) >= MIN_CHARS_FOR_TEXT_PAGE and i not in still_bad, ocr=i in ocr_done)
            for i, t in enumerate(cleaned)
        ]
        n_text = sum(p.has_text for p in pages)
        if n_text == 0:
            garbled = any(len(t.strip()) >= MIN_CHARS_FOR_TEXT_PAGE for t in raw_pages)
            if garbled:
                msg = ("This PDF's text layer is unreadable (its fonts have no character mapping), "
                       "so the text cannot be extracted reliably.")
            else:
                msg = "No text could be extracted. This PDF looks like a scan or image-only document."
            hint = (" OCR was attempted but did not produce readable text." if ocr_available()
                    else " OCR is not available on this server (install Tesseract to enable it).")
            raise EmptyPDFError(msg + hint)
        if n_text < len(pages):
            missing = [p.number for p in pages if not p.has_text]
            shown = ", ".join(map(str, missing[:15])) + ("..." if len(missing) > 15 else "")
            warnings.append(
                f"{len(missing)} page(s) had no readable text (blank, image-only or unreadable fonts) "
                f"and are not covered by the analysis: {shown}."
            )

        meta = {k: v.strip() for k, v in (doc.metadata or {}).items() if isinstance(v, str) and v.strip()}
        try:
            headings = [Heading(normalize_chars(h.text).strip(), h.level, h.page) for h in _detect_headings(doc)]
        except Exception:
            headings = []

        reference_pages: list[int] = []
        if limits.skip_references:
            refs_hint = [p.number for p in pages if p.has_text and _REFS_START.search(p.text)]
            extra: list[Heading] = []
            if refs_hint:   # outlines often omit appendices, so also look at fonts after the references
                try:
                    extra = [Heading(normalize_chars(h.text).strip(), h.level, h.page)
                             for h in _font_headings(doc, refs_hint[-1], 60)]
                except Exception:
                    extra = []
            reference_pages = _mark_references(pages, headings + extra)

        return ExtractedDocument(
            filename=filename,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            page_count=doc.page_count,
            pages=pages,
            metadata=meta,
            headings=headings,
            warnings=warnings,
            reference_pages=reference_pages,
            ocr_pages=[i + 1 for i in ocr_done],
        )
    finally:
        doc.close()
