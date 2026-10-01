"""Turn a validated Report into the JSON the browser renders.

Section bodies are LLM output (and the LLM read an untrusted PDF), so they go through
`safe_markdown`; the page's CSP is a second line of defence.
"""

from __future__ import annotations

from ..reports.exporter import BASIS_LABEL, overview_rows
from ..reports.models import Report
from ..utils.safe_markdown import safe_markdown
from ..utils.text import format_pages

__all__ = ["report_payload", "safe_markdown"]


def report_payload(r: Report) -> dict:
    return {
        "title": r.overview.title,
        "title_source": r.overview.title_source,
        "overview": [{"label": k, "value": v} for k, v in overview_rows(r)],
        "sections": [
            {
                "key": s.key,
                "heading": s.heading,
                "basis": s.basis,
                "basis_label": BASIS_LABEL[s.basis],
                "source": format_pages(s.pages),
                "supported": s.supported,
                "html": safe_markdown(s.body_markdown),
                "evidence": [{"source": format_pages([e.page]), "quote": e.quote} for e in s.evidence],
            }
            for s in r.sections
        ],
        "metrics": [
            {"label": m.label, "value": m.value, "source": format_pages(m.pages), "verified": m.verified}
            for m in r.metrics
        ],
        "not_stated": r.not_stated,
        "references": [
            {"source": format_pages([ref.page]), "used_in": ref.used_in, "quotes": ref.quotes} for ref in r.references
        ],
        "validation_notes": r.validation_notes,
        "warnings": r.warnings,
        "meta": r.meta,
    }
