"""Web API: upload -> job -> report -> export, for two unrelated PDFs, plus error paths."""

import time

import pytest
from fastapi.testclient import TestClient

from researchos.utils.config import Limits
from researchos.web.api import create_app
from researchos.web.render import safe_markdown

from .fake_llm import FakeLLMClient

LIMITS = Limits(max_pdf_mb=5, max_pages=100, chunk_chars=1500, single_pass_chars=3000, notes_budget_chars=6000)


@pytest.fixture()
def client():
    app = create_app(client_factory=FakeLLMClient, limits_factory=lambda: LIMITS)
    with TestClient(app) as c:
        yield c


def run_job(client, pdf: bytes, name: str) -> dict:
    r = client.post("/api/jobs", files={"file": (name, pdf, "application/pdf")})
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_index_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and "ResearchOS" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["max_pdf_mb"] == 5 and cfg["llm_configured"] is True


def test_two_pdfs_give_two_different_reports(client, pdf_a, pdf_b):
    a = run_job(client, pdf_a, "a.pdf")
    b = run_job(client, pdf_b, "b.pdf")
    assert a["status"] == b["status"] == "done"
    assert a["report"]["title"].startswith("Coral") and b["report"]["title"].startswith("Quarterly Warehouse")
    assert a["document"]["pages"] == 4 and b["document"]["pages"] == 3
    text_b = str(b["report"]).lower()
    assert "coral" not in text_b and "zooxanthellae" not in text_b
    md = client.get(f"/api/jobs/{b['id']}/export/md")
    assert md.status_code == 200 and "Quarterly Warehouse" in md.text and "attachment" in md.headers["content-disposition"]
    pdf = client.get(f"/api/jobs/{a['id']}/export/pdf")
    assert pdf.content.startswith(b"%PDF")


def test_upload_errors(client, pdf_a):
    assert client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
    r = client.post("/api/jobs", files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 400 and "not a PDF" in r.json()["error"]
    assert client.post("/api/jobs", files={"file": ("big.pdf", b"%PDF-" + b"0" * (6 * 1024 * 1024), "application/pdf")}).status_code == 413
    broken = run_job(client, pdf_a[: len(pdf_a) // 3], "broken.pdf")
    assert broken["status"] in ("error", "done")
    assert "Traceback" not in str(broken.get("error"))
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_missing_llm_config_is_reported_on_the_job(pdf_a, monkeypatch):
    from researchos.utils.config import LLMConfig
    from researchos.llm.client import LLMClient

    app = create_app(client_factory=lambda: LLMClient(LLMConfig.from_env()), limits_factory=lambda: LIMITS)
    with TestClient(app) as c:
        job = run_job(c, pdf_a, "a.pdf")
    assert job["status"] == "error" and job["error_is_config"]
    assert "GROQ_API_KEY" in job["error"]
    assert job["document"]["pages"] == 4          # extraction facts are still shown


def test_delete_job(client, pdf_a):
    job = run_job(client, pdf_a, "a.pdf")
    assert client.delete(f"/api/jobs/{job['id']}").json()["deleted"] is True
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_markdown_rendering_neutralises_html_and_links():
    html = safe_markdown('Hi <script>alert(1)</script> [x](javascript:alert(1)) ![i](http://evil/x.png) [ok](https://example.org)\n\n| a | b |\n|---|---|\n| 1 | 2 |')
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "javascript:" not in html and "<img" not in html
    assert 'href="https://example.org"' in html and "<table>" in html
