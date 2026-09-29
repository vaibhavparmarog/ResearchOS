"""PDF ingestion: validation, extraction, page provenance, and graceful errors."""

import pytest

from pdf_analyst.pdf.chunker import chunk_document
from pdf_analyst.pdf.extractor import extract_pdf
from pdf_analyst.utils.config import Limits
from pdf_analyst.utils.errors import (
    CorruptPDFError,
    EmptyPDFError,
    FileTooLargeError,
    InvalidFileTypeError,
    NoFileError,
    TooManyPagesError,
)

from .conftest import PDF_A_PAGES
from .pdf_factory import make_image_only_pdf, make_pdf


def test_extracts_pages_with_numbers(pdf_a, limits):
    doc = extract_pdf(pdf_a, "a.pdf", limits)
    assert doc.page_count == len(PDF_A_PAGES)
    assert [p.number for p in doc.pages] == [1, 2, 3, 4]
    assert "Coral Bleaching" in doc.page_text(1)
    assert "87.5%" in doc.page_text(3)          # page 3, not any other page
    assert "87.5%" not in doc.page_text(2)
    assert doc.sha256 and doc.size_bytes == len(pdf_a)


def test_two_pdfs_extract_independently(pdf_a, pdf_b, limits):
    a, b = extract_pdf(pdf_a, "a.pdf", limits), extract_pdf(pdf_b, "b.pdf", limits)
    assert a.sha256 != b.sha256
    assert a.page_count != b.page_count
    assert "Coral" not in " ".join(p.text for p in b.pages)


def test_chunks_keep_page_markers_and_provenance(limits):
    pdf = make_pdf([f"Page body number {i}. " + "Some sentence about the subject. " * 25 for i in range(1, 9)])
    doc = extract_pdf(pdf, "long.pdf", limits)
    chunks = chunk_document(doc, limits)
    assert len(chunks) > 1
    for c in chunks:
        for n in c.pages:
            assert f"[[Page {n}]]" in c.text
            assert f"Page body number {n}." in c.text      # the marker sits next to that page's own text
    assert sorted({n for c in chunks for n in c.pages}) == list(range(1, 9))


def test_oversized_page_is_split_but_keeps_its_page_number(limits):
    doc = extract_pdf(make_pdf(["word " * 300 + "\n\n" + "other " * 300]), "one.pdf", limits)
    chunks = chunk_document(doc, limits)
    assert len(chunks) >= 2 and all(c.pages == [1] for c in chunks)


def test_running_headers_removed(limits):
    topics = ["harbour tides", "grain storage", "bridge cables", "lens grinding", "salt marshes", "tram scheduling"]
    body = "Notes on {t} appear first here.\nA second line of ordinary prose.\nA third line follows.\nAnd a closing line."
    pdf = make_pdf([body.format(t=t) for t in topics], header="ACME CONFIDENTIAL")
    doc = extract_pdf(pdf, "h.pdf", limits)
    assert all("ACME CONFIDENTIAL" not in p.text for p in doc.pages)
    assert "Notes on bridge cables appear first" in doc.page_text(3)


def test_repeated_body_wording_is_not_mistaken_for_a_header(limits):
    body = "Intro line number {i} of a short list.\nItem: alpha value {i}\nItem: beta value {i}\nItem: gamma value {i}\nItem: delta value {i}\nItem: epsilon value {i}\nClosing remark {i}."
    doc = extract_pdf(make_pdf([body.format(i=i) for i in range(1, 7)]), "list.pdf", limits)
    assert "Item: gamma value 4" in doc.page_text(4)


def test_no_file(limits):
    with pytest.raises(NoFileError):
        extract_pdf(None, "x.pdf", limits)
    with pytest.raises(NoFileError):
        extract_pdf(b"", "x.pdf", limits)


def test_invalid_file_type(limits):
    with pytest.raises(InvalidFileTypeError):
        extract_pdf(b"just some text", "notes.txt", limits)
    with pytest.raises(InvalidFileTypeError):          # right extension, wrong content
        extract_pdf(b"just some text", "fake.pdf", limits)


def test_corrupted_pdf(pdf_a, limits):
    corrupted = pdf_a[: len(pdf_a) // 3]
    with pytest.raises((CorruptPDFError, EmptyPDFError)) as ei:
        extract_pdf(corrupted, "broken.pdf", limits)
    assert "Traceback" not in ei.value.user_message
    with pytest.raises(CorruptPDFError):
        extract_pdf(b"%PDF-1.7\n" + b"\x00garbage" * 50, "junk.pdf", limits)


def test_blank_pdf_is_empty(limits):
    with pytest.raises(EmptyPDFError):
        extract_pdf(make_pdf(["", ""]), "blank.pdf", limits)


def test_image_only_pdf_reports_scan(limits):
    with pytest.raises(EmptyPDFError) as ei:
        extract_pdf(make_image_only_pdf(), "scan.pdf", limits)
    assert "scan" in ei.value.user_message.lower()


def test_partially_empty_pdf_warns(limits):
    doc = extract_pdf(make_pdf(["Real text on the first page of this file.", "", "More real text on page three of it."]), "p.pdf", limits)
    assert doc.page_count == 3 and len(doc.text_pages) == 2
    assert any("page 2" in w.lower() or "2" in w for w in doc.warnings)


def test_size_and_page_limits(pdf_a):
    with pytest.raises(FileTooLargeError):
        extract_pdf(pdf_a, "a.pdf", Limits(max_pdf_mb=0))
    with pytest.raises(TooManyPagesError):
        extract_pdf(pdf_a, "a.pdf", Limits(max_pages=2))


def test_encrypted_pdf(limits):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text here")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    from pdf_analyst.utils.errors import EncryptedPDFError

    with pytest.raises(EncryptedPDFError):
        extract_pdf(data, "locked.pdf", limits)
