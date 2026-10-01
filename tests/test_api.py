"""Web API: upload -> job -> report -> export, for two unrelated PDFs, plus privacy, cancel and error paths."""

import time

import pytest
from fastapi.testclient import TestClient

from researchos.utils.config import Limits
from researchos.web.api import create_app
from researchos.web.render import safe_markdown

from .fake_llm import FakeLLMClient

LIMITS = Limits(max_pdf_mb=5, max_pages=100, chunk_chars=1500, single_pass_chars=3000, notes_budget_chars=6000)


def make_client(**fake_kw):
    app = create_app(client_factory=lambda cancel: FakeLLMClient(cancel_event=cancel, **fake_kw), limits_factory=lambda: LIMITS)
    return TestClient(app)


@pytest.fixture()
def client():
    with make_client() as c:
        yield c


def upload(client, pdf: bytes, name: str):
    return client.post("/api/jobs", files={"file": (name, pdf, "application/pdf")})


def wait(client, job_id: str, until=("done", "error", "cancelled")) -> dict:
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job.get("status") in until:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def run_job(client, pdf: bytes, name: str) -> dict:
    r = upload(client, pdf, name)
    assert r.status_code == 202, r.text
    return wait(client, r.json()["id"])


def test_index_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and "ResearchOS" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config")
    assert cfg.json()["max_pdf_mb"] == 5 and cfg.json()["llm_configured"] is True
    assert cfg.headers["cache-control"] == "no-store"


def test_two_pdfs_give_two_different_reports_and_stateless_exports(client, pdf_a, pdf_b):
    a = run_job(client, pdf_a, "a.pdf")
    b = run_job(client, pdf_b, "b.pdf")
    assert a["status"] == b["status"] == "done"
    assert a["report"]["title"].startswith("Coral") and b["report"]["title"].startswith("Quarterly Warehouse")
    assert a["document"]["pages"] == 4 and b["document"]["pages"] == 3
    text_b = str(b["report"]).lower()
    assert "coral" not in text_b and "zooxanthellae" not in text_b
    # exports are rendered from the copy the browser sends back
    md = client.post("/api/export/md", json=b["report_data"])
    assert md.status_code == 200 and "Quarterly Warehouse" in md.text and "attachment" in md.headers["content-disposition"]
    pdf = client.post("/api/export/pdf", json=a["report_data"])
    assert pdf.content.startswith(b"%PDF")
    assert client.post("/api/export/md", json={"nope": 1}).status_code == 400
    assert client.post("/api/export/exe", json=a["report_data"]).status_code == 400


def test_report_is_purged_after_delivery(pdf_a):
    with make_client() as c:
        jobs = c.app.state.jobs
        jobs._keep_delivered = 0.0
        job = run_job(c, pdf_a, "a.pdf")
        assert job["status"] == "done"
        for _ in range(80):                         # reaper runs every 5 s; do its sweep here instead
            time.sleep(0.05)
            for j in list(jobs._jobs.values()):
                if j.delivered is not None:
                    jobs.cancel(j.id)
        assert c.get(f"/api/jobs/{job['id']}").status_code == 404
        assert not jobs._jobs                       # nothing left in memory


def test_extracted_text_is_not_kept(client, pdf_a):
    job = run_job(client, pdf_a, "a.pdf")
    stored = client.app.state.jobs._jobs[job["id"]]
    assert not hasattr(stored, "document") or getattr(stored, "document", None) is None
    assert stored.doc_info["pages"] == 4


def test_cancel_stops_a_running_job(pdf_a):
    with make_client(delay=5.0) as c:
        r = upload(c, pdf_a, "a.pdf")
        job_id = r.json()["id"]
        wait(c, job_id, until=("analyzing",))
        assert c.post(f"/api/jobs/{job_id}/cancel").json()["cancelled"] is True
        assert c.get(f"/api/jobs/{job_id}").status_code == 404
        assert c.post(f"/api/jobs/{job_id}/cancel").json()["cancelled"] is False


def test_too_many_active_jobs_per_client(pdf_a):
    with make_client(delay=5.0) as c:
        ids = [upload(c, pdf_a, "a.pdf").json()["id"] for _ in range(2)]
        third = upload(c, pdf_a, "a.pdf")
        assert third.status_code == 429 and "already" in third.json()["error"]
        for i in ids:
            c.post(f"/api/jobs/{i}/cancel")


def test_upload_errors(client, pdf_a):
    assert client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
    r = client.post("/api/jobs", files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 400 and "not a PDF" in r.json()["error"]
    assert client.post("/api/jobs", files={"file": ("big.pdf", b"%PDF-" + b"0" * (6 * 1024 * 1024), "application/pdf")}).status_code == 413
    broken = run_job(client, pdf_a[: len(pdf_a) // 3], "broken.pdf")
    assert broken["status"] in ("error", "done")
    assert "Traceback" not in str(broken.get("error"))
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_missing_llm_config_is_reported_on_the_job(pdf_a):
    from researchos.llm.client import LLMClient
    from researchos.utils.config import LLMConfig

    app = create_app(client_factory=lambda cancel: LLMClient(LLMConfig.from_env(), cancel_event=cancel), limits_factory=lambda: LIMITS)
    with TestClient(app) as c:
        job = run_job(c, pdf_a, "a.pdf")
    assert job["status"] == "error" and job["error_is_config"]
    assert "GROQ_API_KEY" in job["error"]
    assert job["document"]["pages"] == 4          # extraction facts are still shown


def test_markdown_rendering_neutralises_html_and_links():
    html = safe_markdown('Hi <script>alert(1)</script> [x](javascript:alert(1)) ![i](http://evil/x.png) [ok](https://example.org)\n\n| a | b |\n|---|---|\n| 1 | 2 |')
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "javascript:" not in html and "<img" not in html
    assert 'href="https://example.org"' in html and "<table>" in html


def test_pdf_export_ignores_injected_html(client, pdf_a):
    job = run_job(client, pdf_a, "a.pdf")
    data = job["report_data"]
    data["sections"][0]["body_markdown"] = '<img src="/etc/passwd"> <iframe src="file:///etc/passwd"></iframe> ok'
    r = client.post("/api/export/pdf", json=data)
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
