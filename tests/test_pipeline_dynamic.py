"""Proof that the report depends on the uploaded PDF: A then B (then C) yield different, correctly sourced reports."""

import re

from pdf_analyst.pdf.extractor import extract_pdf
from pdf_analyst.reports.exporter import to_markdown
from pdf_analyst.reports.generator import generate_report
from pdf_analyst.utils.text import match_form

from .conftest import PDF_A_PAGES, PDF_B_PAGES
from .fake_llm import FakeLLMClient
from .pdf_factory import make_pdf

A_ONLY = ["coral", "zooxanthellae", "staghorn", "solvik", "reef"]
B_ONLY = ["warehouse", "forklift", "haldorsen", "rotterdam", "pallet"]


def run(pdf, name, limits, client=None):
    doc = extract_pdf(pdf, name, limits)
    client = client or FakeLLMClient()
    return doc, client, generate_report(doc, client, limits)


def all_text(report) -> str:
    return to_markdown(report).lower()


def test_report_a_is_about_a(pdf_a, limits):
    doc, client, rep = run(pdf_a, "a.pdf", limits)
    assert rep.overview.title == "Coral Bleaching Resilience in the Outer Reef"
    assert rep.overview.page_count == 4
    text = all_text(rep)
    assert any(w in text for w in A_ONLY)
    assert not any(w in text for w in B_ONLY)


def test_a_then_b_reports_differ_and_do_not_leak(pdf_a, pdf_b, limits):
    doc_a, client_a, rep_a = run(pdf_a, "a.pdf", limits)
    # SAME client object reused for the second upload: nothing may carry over.
    doc_b, client_b, rep_b = run(pdf_b, "b.pdf", limits, client=client_a)

    assert rep_a.overview.title != rep_b.overview.title
    assert rep_b.overview.title == "Quarterly Warehouse Logistics Review"
    assert rep_b.overview.page_count == 3 and rep_a.overview.page_count == 4
    assert to_markdown(rep_a) != to_markdown(rep_b)

    text_b = all_text(rep_b)
    assert any(w in text_b for w in B_ONLY)
    assert not any(w in text_b for w in A_ONLY), "content of PDF A leaked into report B"

    # Prompts sent while processing B must contain nothing from A.
    n_a = len(client_a.prompts) - client_b.calls + 0
    b_prompts = " ".join(u for _, u in client_a.prompts[rep_a.meta["llm_calls"]:]).lower()
    assert b_prompts and not any(w in b_prompts for w in A_ONLY)
    a_prompts = " ".join(u for _, u in client_a.prompts[: rep_a.meta["llm_calls"]]).lower()
    assert not any(w in a_prompts for w in B_ONLY)


def test_evidence_and_pages_point_into_the_right_pdf(pdf_a, pdf_b, limits):
    for pdf, name in ((pdf_a, "a.pdf"), (pdf_b, "b.pdf")):
        doc, _, rep = run(pdf, name, limits)
        checked = 0
        for s in rep.sections:
            for p in s.pages:
                assert 1 <= p <= doc.page_count
            for e in s.evidence:
                assert match_form(e.quote) in match_form(doc.page_text(e.page)), "quote not on the cited page"
                checked += 1
        assert checked > 0
        for ref in rep.references:
            assert 1 <= ref.page <= doc.page_count
        for m in rep.metrics:
            assert m.verified and all(match_form(m.value) in match_form(doc.page_text(p)) for p in m.pages)


def test_metrics_come_from_the_uploaded_document(pdf_a, pdf_b, limits):
    _, _, rep_a = run(pdf_a, "a.pdf", limits)
    _, _, rep_b = run(pdf_b, "b.pdf", limits)
    vals_a, vals_b = {m.value for m in rep_a.metrics}, {m.value for m in rep_b.metrics}
    assert vals_a and vals_b and vals_a.isdisjoint(vals_b)


def test_long_pdf_uses_map_reduce_and_keeps_page_provenance(limits):
    # 40 pages; each page has a unique token so we can verify provenance per page.
    pages = [
        f"Section {i} of the handbook. The protocol number P{i:03d} governs handling of item class K{i:03d}. "
        f"Measured throughput in this section was {i * 7 + 100} units per hour, which the authors note is stable."
        for i in range(1, 41)
    ]
    pages[0] = "Field Handbook For Item Handling\nEdited by Corwin Bexley\n2020\n\n" + pages[0]
    doc, client, rep = run(make_pdf(pages), "long.pdf", limits)
    assert rep.meta["strategy"] == "map-reduce"
    assert rep.meta["chunks"] > 3
    assert client.calls > rep.meta["chunks"]            # chunk calls + condense/final calls
    assert rep.overview.title == "Field Handbook For Item Handling"
    for s in rep.sections:
        for e in s.evidence:
            assert match_form(e.quote) in match_form(doc.page_text(e.page))
    # Every chunk prompt only contains pages that chunk owns
    for system, user in client.prompts:
        if "TASK: CHUNK_NOTES" in system:
            pages_in = {int(n) for n in re.findall(r"\[\[Page (\d+)\]\]", user)}
            for n in pages_in:
                assert f"P{n:03d}" in user or n == 1
    cited = {p for s in rep.sections for p in s.pages}
    assert len(cited) > 3 and cited <= set(range(1, 41))


def test_fit_notes_bounds_size_so_reduction_converges():
    from pdf_analyst.analysis.synthesizer import _size, fit_notes

    big = {"summary": "s" * 50, "topics": [], "concepts": [{"term": "t", "explanation": "e" * 200, "pages": [1]}] * 30,
           "points": [{"text": "p" * 300, "kind": "claim", "pages": [1]}] * 60, "data": [], "limitations_stated": []}
    small = fit_notes(big, 3000)
    assert _size(small) <= 3000 and small["points"] and small["concepts"]


def test_second_upload_of_same_bytes_is_reproducible_and_first_never_reused(pdf_a, pdf_b, limits):
    _, c1, r1 = run(pdf_a, "a.pdf", limits)
    _, c2, r2 = run(pdf_a, "a.pdf", limits)
    assert r1.sections[0].body_markdown == r2.sections[0].body_markdown
    # no module-level state: a different PDF right after gives a different first section
    _, _, r3 = run(pdf_b, "b.pdf", limits)
    assert r3.sections[0].body_markdown != r1.sections[0].body_markdown
