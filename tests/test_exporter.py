"""Exports must contain the report for the CURRENT document."""

import pymupdf

from pdf_analyst.pdf.extractor import extract_pdf
from pdf_analyst.reports import exporter
from pdf_analyst.reports.generator import generate_report

from .fake_llm import FakeLLMClient


def report_for(pdf, name, limits):
    return generate_report(extract_pdf(pdf, name, limits), FakeLLMClient(), limits)


def test_markdown_txt_json_reflect_current_document(pdf_a, pdf_b, limits):
    ra, rb = report_for(pdf_a, "a.pdf", limits), report_for(pdf_b, "b.pdf", limits)
    for rep, title, other in ((ra, "Coral Bleaching", "Warehouse"), (rb, "Quarterly Warehouse", "Coral")):
        for render in (exporter.to_markdown, exporter.to_text, exporter.to_json):
            out = render(rep)
            assert title in out and other not in out
    md = exporter.to_markdown(ra)
    assert "## Source References" in md and "Page 1" in md
    assert "#" not in exporter.to_text(ra).split("\n")[0]        # markup stripped


def test_pdf_export_is_a_valid_pdf_with_the_report_text(pdf_b, limits):
    rep = report_for(pdf_b, "b.pdf", limits)
    data = exporter.to_pdf(rep)
    assert data.startswith(b"%PDF")
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        text = "\n".join(p.get_text() for p in doc)
    assert "Quarterly Warehouse Logistics Review" in text and "Source References" in text
