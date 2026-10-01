"""Turn a validated Report into the JSON the browser renders.

Section bodies are LLM output, and the LLM read an untrusted PDF, so Markdown is rendered with
raw HTML disabled, links restricted to http(s), and images stripped. The page's CSP is a second
line of defence.
"""

from __future__ import annotations

import re

import markdown

from ..reports.exporter import BASIS_LABEL, overview_rows
from ..reports.models import Report
from ..utils.text import format_pages

_HREF = re.compile(r'href="([^"]*)"')
_IMG = re.compile(r"<img\b[^>]*>", re.I)


def safe_markdown(text: str) -> str:
    md = markdown.Markdown(extensions=["tables", "sane_lists", "fenced_code"])
    md.preprocessors.deregister("html_block")     # raw HTML is shown as text, never interpreted
    md.inlinePatterns.deregister("html")
    html = md.convert(text or "")
    html = _IMG.sub("", html)

    def fix(m: re.Match) -> str:
        url = m.group(1)
        if re.match(r"^(https?://|#)", url, re.I):
            return f'href="{url}" rel="noopener noreferrer nofollow" target="_blank"'
        return 'href="#"'

    return _HREF.sub(fix, html)


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
