"""Markdown -> HTML for untrusted text (LLM output that read an untrusted PDF).

Raw HTML is shown as text, images are removed (no local-file or remote loads), and links are
limited to http(s)/anchors. Used for the web page and for the PDF export.
"""

from __future__ import annotations

import re

import markdown

_HREF = re.compile(r'href="([^"]*)"')
_IMG = re.compile(r"<img\b[^>]*>", re.I)


def safe_markdown(text: str) -> str:
    md = markdown.Markdown(extensions=["tables", "sane_lists", "fenced_code"])
    md.preprocessors.deregister("html_block")     # raw HTML is escaped, never interpreted
    md.inlinePatterns.deregister("html")
    html = _IMG.sub("", md.convert(text or ""))

    def fix(m: re.Match) -> str:
        url = m.group(1)
        if re.match(r"^(https?://|#)", url, re.I):
            return f'href="{url}" rel="noopener noreferrer nofollow" target="_blank"'
        return 'href="#"'

    return _HREF.sub(fix, html)
