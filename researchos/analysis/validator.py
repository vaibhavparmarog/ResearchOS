"""Grounding checks: every page reference and quote in LLM output is checked against the real PDF.

Nothing the model says about *where* something comes from is trusted:
- page numbers must exist (and have text) in the uploaded PDF, else they are dropped;
- quotes must occur verbatim on the cited page, else the page is corrected if the quote
  is found elsewhere in the document, or the quote is dropped;
- metrics must be findable in the document, else they are flagged as unverified;
- title/authors/date must appear in the opening pages or PDF metadata, else they are dropped.
"""

from __future__ import annotations

import re
from typing import Any

from ..llm.prompts import SECTION_KEYS
from ..pdf.extractor import ExtractedDocument
from ..reports.models import Evidence, Metric, Overview, Report, Section, SourceRef
from ..utils.errors import LLMResponseError
from ..utils.text import match_form

MIN_QUOTE_CHARS = 15
MAX_QUOTE_CHARS = 300
PAGE_MENTION = re.compile(r"\b(?:pages?|pp?\.)\s*(\d{1,5})(?:\s*[-–—]\s*(\d{1,5}))?", re.I)
PAGE_MARKER = re.compile(r"\[\[Page (\d+)\]\]")


class PageIndex:
    """Normalised page texts for verification (built once per document)."""

    def __init__(self, doc: ExtractedDocument):
        self.doc = doc
        self.norm = {p.number: match_form(p.text) for p in doc.pages if p.has_text}
        self.valid = set(self.norm)

    def coerce_pages(self, raw: Any) -> tuple[list[int], int]:
        """Return (valid sorted unique pages, number of invalid references dropped)."""
        if raw is None:
            return [], 0
        if not isinstance(raw, (list, tuple, set)):
            raw = [raw]
        out: set[int] = set()
        dropped = 0
        for item in raw:
            nums: list[int] = []
            if isinstance(item, bool):
                dropped += 1
                continue
            if isinstance(item, int):
                nums = [item]
            elif isinstance(item, float) and item.is_integer():
                nums = [int(item)]
            elif isinstance(item, str):
                m = re.fullmatch(r"\s*(?:p{1,2}\.?\s*)?(\d+)\s*(?:[-–—]\s*(\d+))?\s*", item, re.I)
                if m:
                    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
                    nums = list(range(a, b + 1)) if a <= b and b - a <= 50 else []
            if not nums:
                dropped += 1
                continue
            for n in nums:
                if n in self.valid:
                    out.add(n)
                else:
                    dropped += 1
        return sorted(out), dropped

    def quote_on_page(self, quote: str, page: int) -> bool:
        needle = match_form(quote)
        return len(needle) >= MIN_QUOTE_CHARS and needle in self.norm.get(page, "")

    def find_quote(self, quote: str, prefer: list[int]) -> int | None:
        """Page where the quote occurs verbatim; prefers the declared/nearby pages."""
        needle = match_form(quote)
        if len(needle) < MIN_QUOTE_CHARS:
            return None
        for page in [*prefer, *sorted(self.norm)]:
            if needle in self.norm.get(page, ""):
                return page
        return None

    def text_anywhere(self, needle: str, pages: list[int] | None = None) -> list[int]:
        """Pages (within `pages`, default all) whose text contains `needle`.

        If the needle has digits, a page also matches when every number in it appears as a whole
        token (tolerates "94.2 %" vs "94.2%" and reformatted units).
        """
        n = match_form(needle)
        if not n:
            return []
        scope = pages if pages else sorted(self.norm)
        nums = re.findall(r"\d[\d,]*(?:\.\d+)?", n)
        hits = []
        for p in scope:
            text = self.norm.get(p, "")
            if n in text or (nums and all(re.search(rf"(?<![\d.,]){re.escape(x)}(?![\d]|[.,]\d)", text) for x in nums)):
                hits.append(p)
        return hits


# --------------------------------------------------------------------------- helpers
def _str(value: Any, limit: int = 20000) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _verified_evidence(
    raw: Any, idx: PageIndex, section_pages: list[int], notes: list[str], heading: str
) -> list[Evidence]:
    out: list[Evidence] = []
    seen: set[str] = set()
    for item in _list(raw):
        if not isinstance(item, dict):
            continue
        quote = _str(item.get("quote"), MAX_QUOTE_CHARS)
        pages, _ = idx.coerce_pages(item.get("page"))
        declared = pages[:1]
        if not quote or quote in seen:
            continue
        page = idx.find_quote(quote, [*declared, *section_pages])
        if page is None:
            notes.append(f"Removed a quote in “{heading}” that does not appear in the PDF.")
            continue
        if item.get("page") in (None, "", []):
            notes.append(f"Added the correct page ({page}) to an uncited quote in “{heading}”.")
        elif not declared or page != declared[0]:
            was = declared[0] if declared else item.get("page")
            notes.append(f"Corrected a quote's page in “{heading}” from {was} to {page}.")
        seen.add(quote)
        out.append(Evidence(page=page, quote=quote))
    return out


def _clean_inline_pages(text: str, idx: PageIndex) -> tuple[str, int]:
    """Replace [[Page N]] markers with (p. N) and remove inline page mentions that do not exist."""
    text = PAGE_MARKER.sub(lambda m: f"(p. {m.group(1)})", text)
    removed = 0

    def check(m: re.Match) -> str:
        nonlocal removed
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a in idx.valid and b in idx.valid and a <= b:
            return m.group(0)
        removed += 1
        return "[unverified page reference removed]"

    return PAGE_MENTION.sub(check, text), removed


# --------------------------------------------------------------------------- notes (map stage)
def sanitize_notes(notes: dict, chunk_pages: list[int], idx: PageIndex) -> dict:
    """Restrict a chunk's notes to pages that chunk actually contained; drop unverifiable quotes."""
    allowed = set(chunk_pages)

    def pages_of(item: dict) -> list[int]:
        pages, _ = idx.coerce_pages(item.get("pages"))
        return [p for p in pages if p in allowed]

    def dicts(key: str) -> list[dict]:
        return [i for i in _list(notes.get(key)) if isinstance(i, dict)]

    points = []
    for p in dicts("points"):
        text = _str(p.get("text"), 600)
        if not text:
            continue
        pages = pages_of(p)
        quote = _str(p.get("quote"), MAX_QUOTE_CHARS)
        if quote:
            found = idx.find_quote(quote, pages or chunk_pages)
            if found is None or found not in allowed:
                quote = ""
            elif not pages:
                pages = [found]
        if not pages:
            continue
        points.append({"text": text, "kind": _str(p.get("kind"), 20) or "other", "pages": pages, **({"quote": quote} if quote else {})})

    concepts = [
        {"term": _str(c.get("term"), 120), "explanation": _str(c.get("explanation"), 500), "pages": pages_of(c)}
        for c in dicts("concepts") if _str(c.get("term")) and pages_of(c)
    ]
    data = [
        {"label": _str(d.get("label"), 200), "value": _str(d.get("value"), 120), "context": _str(d.get("context"), 300), "pages": pages_of(d)}
        for d in dicts("data") if _str(d.get("label")) and _str(d.get("value")) and pages_of(d)
    ]
    lims = [{"text": _str(l.get("text"), 500), "pages": pages_of(l)} for l in dicts("limitations_stated") if _str(l.get("text")) and pages_of(l)]
    return {
        "summary": _str(notes.get("summary"), 1500),
        "topics": [_str(t, 80) for t in _list(notes.get("topics")) if isinstance(t, str) and t.strip()][:12],
        "points": points[:40],
        "concepts": concepts[:25],
        "data": data[:30],
        "limitations_stated": lims[:15],
        "pages_covered": sorted(allowed),
    }


# --------------------------------------------------------------------------- final report
def _verify_overview(raw: Any, doc: ExtractedDocument, idx: PageIndex, notes: list[str]) -> Overview:
    raw = raw if isinstance(raw, dict) else {}
    opening = [p.number for p in doc.text_pages[:3]]
    meta_blob = " ".join(doc.metadata.get(k, "") for k in ("title", "author", "creationDate", "subject"))
    opening_norm = " ".join(idx.norm.get(p, "") for p in opening) + " " + match_form(meta_blob)

    title = _str(raw.get("title"), 300)
    title_source = "document"
    if title and match_form(title) not in opening_norm:
        notes.append("Discarded the generated title because it does not appear in the PDF's opening pages or metadata.")
        title = ""
    if not title:
        meta_title = doc.metadata.get("title", "")
        if meta_title and len(meta_title) > 3:
            title, title_source = meta_title, "pdf_metadata"
        else:
            title, title_source = re.sub(r"\.pdf$", "", doc.filename, flags=re.I), "filename"

    authors = []
    for a in _list(raw.get("authors")):
        a = _str(a, 120)
        if not a:
            continue
        if match_form(a) in opening_norm:
            authors.append(a)
        else:
            notes.append(f"Removed author “{a}”: not found in the PDF's opening pages or metadata.")

    date = _str(raw.get("date"), 60) or None
    if date:
        years = re.findall(r"\d{4}", date)
        if years and all(y in opening_norm for y in years):
            pass
        elif years and idx.text_anywhere(years[0]):
            pass
        else:
            notes.append(f"Removed date “{date}”: not found in the PDF.")
            date = None

    return Overview(
        title=title,
        title_source=title_source,
        authors=authors,
        date=date,
        document_type=_str(raw.get("document_type"), 80) or "Not determined",
        page_count=doc.page_count,
        filename=doc.filename,
        size_bytes=doc.size_bytes,
    )


def _verify_metrics(raw: Any, idx: PageIndex, notes: list[str]) -> list[Metric]:
    metrics: list[Metric] = []
    for m in _list(raw)[:12]:
        if not isinstance(m, dict):
            continue
        label, value = _str(m.get("label"), 200), _str(m.get("value"), 120)
        if not label or not value:
            continue
        pages, _ = idx.coerce_pages(m.get("pages"))
        hits = idx.text_anywhere(value, pages)
        if not hits and pages:
            # cited pages are wrong; try to relocate anywhere in the document
            hits = idx.text_anywhere(value)
            if hits:
                notes.append(f"Corrected pages for metric “{label}” to {hits[:3]}.")
        if hits:
            metrics.append(Metric(label, value, hits[:4], True))
        else:
            metrics.append(Metric(label, value, pages, False))
            notes.append(f"Metric “{label}” (value {value}) could not be found in the PDF; marked unverified.")
    return metrics


def validate_report(raw: dict, doc: ExtractedDocument, meta: dict | None = None) -> Report:
    """Turn raw LLM output into a Report whose sources have all been checked against the PDF."""
    idx = PageIndex(doc)
    notes: list[str] = []
    overview = _verify_overview(raw.get("overview"), doc, idx, notes)

    sections: list[Section] = []
    invalid_pages_total = 0
    for s in _list(raw.get("sections")):
        if not isinstance(s, dict):
            continue
        body = _str(s.get("body_markdown"))
        heading = _str(s.get("heading"), 120)
        key = _str(s.get("key"), 40)
        if not body:
            continue
        key = key if key in SECTION_KEYS else "other"
        heading = heading or key.replace("_", " ").title()
        pages, dropped = idx.coerce_pages(s.get("pages"))
        invalid_pages_total += dropped
        evidence = _verified_evidence(s.get("evidence"), idx, pages, notes, heading)
        pages = sorted({*pages, *(e.page for e in evidence)})
        body, removed = _clean_inline_pages(body, idx)
        invalid_pages_total += removed

        basis = _str(s.get("basis"), 20).lower()
        if basis not in ("document", "interpretation", "mixed"):
            basis = "mixed"
        if key == "limitations_inferred":
            basis = "interpretation"
        elif key == "limitations_stated":
            basis = "document"
        supported = not (basis != "interpretation" and not pages)
        if not supported:
            notes.append(f"Section “{heading}” makes document claims but cites no valid page; flagged as unsupported.")
        sections.append(Section(key, heading, basis, body, pages, evidence, supported))

    if not sections:
        raise LLMResponseError(detail="model returned no usable sections")
    order = {k: i for i, k in enumerate([*SECTION_KEYS, "other"])}
    sections.sort(key=lambda s: order[s.key])

    if invalid_pages_total:
        notes.insert(0, f"Removed {invalid_pages_total} page reference(s) that do not exist in this {doc.page_count}-page PDF.")

    metrics = _verify_metrics(raw.get("metrics"), idx, notes)

    refs: dict[int, SourceRef] = {}
    for s in sections:
        for p in s.pages:
            refs.setdefault(p, SourceRef(p)).used_in.append(s.heading)
        for e in s.evidence:
            r = refs.setdefault(e.page, SourceRef(e.page))
            if e.quote not in r.quotes:
                r.quotes.append(e.quote)
    for r in refs.values():   # drop quotes that are merely a fragment of a longer quote on the same page
        norm = {q: match_form(q) for q in r.quotes}
        r.quotes = [q for q in r.quotes if not any(q != o and norm[q] in norm[o] for o in r.quotes)]
    for m in metrics:
        if m.verified:
            for p in m.pages:
                r = refs.setdefault(p, SourceRef(p))
                if "Important data / metrics" not in r.used_in:
                    r.used_in.append("Important data / metrics")

    not_stated = [x for x in (_str(i, 120) for i in _list(raw.get("not_stated"))) if x]
    if not overview.authors and "authors" not in " ".join(not_stated).lower():
        not_stated.append("Authors")
    if not overview.date and "date" not in " ".join(not_stated).lower():
        not_stated.append("Date / year")

    return Report(
        overview=overview,
        sections=sections,
        metrics=metrics,
        references=[refs[p] for p in sorted(refs)],
        not_stated=not_stated,
        validation_notes=notes,
        warnings=list(doc.warnings),
        meta=dict(meta or {}),
    )
