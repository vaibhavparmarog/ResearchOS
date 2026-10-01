"""The validator must neutralise fabricated pages, quotes, authors and metrics."""

from researchos.pdf.extractor import extract_pdf
from researchos.reports.exporter import to_markdown
from researchos.reports.generator import generate_report
from researchos.utils.text import match_form

from .fake_llm import FakeLLMClient


def hallucinating_report(pdf, limits):
    doc = extract_pdf(pdf, "a.pdf", limits)
    return doc, generate_report(doc, FakeLLMClient(hallucinate=True), limits)


def test_invalid_pages_removed(pdf_a, limits):
    doc, rep = hallucinating_report(pdf_a, limits)
    for s in rep.sections:
        assert all(1 <= p <= doc.page_count for p in s.pages)
        assert "999" not in s.body_markdown and "1000" not in s.body_markdown
    assert all(1 <= r.page <= doc.page_count for r in rep.references)
    assert any("page reference" in n for n in rep.validation_notes)


def test_invented_quote_dropped_and_wrong_page_corrected(pdf_a, limits):
    doc, rep = hallucinating_report(pdf_a, limits)
    md = to_markdown(rep)
    assert "invented by the model" not in md
    for s in rep.sections:
        for e in s.evidence:
            assert match_form(e.quote) in match_form(doc.page_text(e.page))
    assert any("Corrected a quote's page" in n for n in rep.validation_notes)
    assert any("does not appear in the PDF" in n for n in rep.validation_notes)


def test_invented_author_and_metric(pdf_a, limits):
    _, rep = hallucinating_report(pdf_a, limits)
    assert "Nonexistent" not in rep.overview.authors and not any("Nonexistent" in a for a in rep.overview.authors)
    invented = [m for m in rep.metrics if m.value == "99.987%"]
    assert not invented or not invented[0].verified          # dropped by fake's [:6] or flagged
    assert all(m.verified for m in rep.metrics if m.value != "99.987%")


def test_unsupported_document_claim_is_flagged(pdf_a, limits):
    _, rep = hallucinating_report(pdf_a, limits)
    meth = next(s for s in rep.sections if s.key == "methodology")
    assert meth.supported is False
    assert "No verifiable page reference" in to_markdown(rep)


def test_interpretation_is_labelled(pdf_a, limits):
    _, rep = hallucinating_report(pdf_a, limits)
    lim = next(s for s in rep.sections if s.key == "limitations_inferred")
    assert lim.basis == "interpretation" and lim.supported
    assert "AI interpretation" in to_markdown(rep)


def test_missing_info_is_stated_not_invented(pdf_b, limits):
    doc = extract_pdf(pdf_b, "b.pdf", limits)
    rep = generate_report(doc, FakeLLMClient(), limits)
    assert rep.overview.date is None or rep.overview.date in doc.page_text(1)
    assert "Not stated in the document" in to_markdown(rep)
