"""LLM configuration + client behaviour, incl. the real HTTP path against the local test double."""

import httpx
import pytest

from pdf_analyst.llm.client import LLMClient, extract_json
from pdf_analyst.pdf.extractor import extract_pdf
from pdf_analyst.reports.generator import generate_report
from pdf_analyst.utils.config import LLMConfig
from pdf_analyst.utils.errors import (
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

from .fake_llm_server import FakeLLMServer


def cfg(server, **kw):
    return LLMConfig(api_key="test-key", base_url=server.base_url, model="m", max_retries=kw.pop("max_retries", 2), **kw)


@pytest.fixture()
def server():
    s = FakeLLMServer().start()
    yield s
    s.shutdown()


def test_missing_api_key_is_a_configuration_error():
    with pytest.raises(LLMConfigError) as ei:
        LLMConfig.from_env()
    msg = ei.value.user_message
    assert "LLM_API_KEY" in msg and "LLM_MODEL" in msg and "Traceback" not in msg


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://example.test/v1/")
    c = LLMConfig.from_env()
    assert (c.api_key, c.model, c.base_url) == ("k", "some-model", "http://example.test/v1")


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_end_to_end_through_real_http_client(pdf_a, pdf_b, limits, server):
    client = LLMClient(cfg(server))
    ra = generate_report(extract_pdf(pdf_a, "a.pdf", limits), client, limits)
    rb = generate_report(extract_pdf(pdf_b, "b.pdf", limits), client, limits)
    assert ra.overview.title.startswith("Coral") and rb.overview.title.startswith("Quarterly Warehouse")
    sent = " ".join(m["content"] for body in server.bodies for m in body["messages"]).lower()
    assert "coral" in sent and "warehouse" in sent      # both uploaded documents were actually sent


def test_rate_limit_is_retried_then_succeeds(server):
    server.fail_first = 2
    client = LLMClient(cfg(server, max_retries=3))
    system = "TASK: CHUNK_NOTES"
    user = "<document_excerpt>\n[[Page 1]]\nA sentence long enough to be treated as a point here.\n</document_excerpt>"
    assert "points" in client.complete_json(system, user)
    assert server.request_count == 3


def test_persistent_rate_limit_raises_friendly_error(server):
    server.fail_first = 99
    with pytest.raises(LLMRateLimitError) as ei:
        LLMClient(cfg(server, max_retries=1)).complete_json("TASK: CHUNK_NOTES", "x")
    assert "rate" in ei.value.user_message.lower()


def test_server_error_and_auth_error(server):
    server.fail_first, server.fail_status = 99, 500
    with pytest.raises(LLMError):
        LLMClient(cfg(server, max_retries=0)).complete_json("TASK: CHUNK_NOTES", "x")
    server.fail_status = 401
    with pytest.raises(LLMConfigError) as ei:
        LLMClient(cfg(server, max_retries=0)).complete_json("TASK: CHUNK_NOTES", "x")
    assert "LLM_API_KEY" in ei.value.user_message


def test_malformed_json_raises_response_error(server):
    # unknown TASK makes the double return non-JSON text, also for the repair attempt
    with pytest.raises(LLMResponseError) as ei:
        LLMClient(cfg(server)).complete_json("TASK: SOMETHING_ELSE", "x")
    assert "Traceback" not in ei.value.user_message


def test_timeout_is_mapped(server, monkeypatch):
    client = LLMClient(cfg(server, max_retries=0))

    def boom(*a, **k):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(client._http, "post", boom)
    with pytest.raises(LLMTimeoutError):
        client.complete_json("TASK: CHUNK_NOTES", "x")


def test_unreachable_service_is_friendly():
    c = LLMClient(LLMConfig(api_key="k", base_url="http://127.0.0.1:9/v1", model="m", max_retries=0, timeout=2))
    with pytest.raises(LLMError) as ei:
        c.complete_json("TASK: CHUNK_NOTES", "x")
    assert "LLM_BASE_URL" in ei.value.user_message
