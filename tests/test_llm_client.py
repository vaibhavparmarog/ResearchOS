"""LLM configuration + client behaviour, incl. the real HTTP path against the local test double."""

import httpx
import pytest

from researchos.llm.client import LLMClient, extract_json
from researchos.pdf.extractor import extract_pdf
from researchos.reports.generator import generate_report
from researchos.utils.config import LLMConfig
from researchos.utils.errors import (
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

from .fake_llm_server import FakeLLMServer


def cfg(server, **kw):
    return LLMConfig.single("test-key", server.base_url, "m", max_retries=kw.pop("max_retries", 2), retry_window=kw.pop("retry_window", 0.3), **kw)


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
    p = c.providers[0]
    assert (p.api_key, p.model, p.base_url) == ("k", "some-model", "http://example.test/v1")


def test_named_providers_and_order(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("NVIDIA_API_KEY", "n")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    c = LLMConfig.from_env()
    assert [p.name for p in c.providers] == ["groq", "openrouter", "nvidia"]
    assert c.providers[0].base_url == "https://api.groq.com/openai/v1"
    assert c.providers[1].base_url == "https://openrouter.ai/api/v1"
    assert (c.providers[2].base_url, c.providers[2].model) == ("https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct")
    monkeypatch.setenv("LLM_PROVIDERS", "nvidia,groq")
    assert [p.name for p in LLMConfig.from_env().providers] == ["nvidia", "groq"]


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
    client = LLMClient(cfg(server, max_retries=3, retry_window=30))
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
    c = LLMClient(LLMConfig.single("k", "http://127.0.0.1:9/v1", "m", max_retries=0, timeout=2))
    with pytest.raises(LLMError) as ei:
        c.complete_json("TASK: CHUNK_NOTES", "x")
    assert "base URL" in ei.value.user_message


# ---- multi-provider failover --------------------------------------------------------------
def two_providers(first, second, **kw):
    from researchos.utils.config import ProviderConfig

    return LLMConfig(
        providers=(
            ProviderConfig("groq", "k1", first.base_url, "m1", key_env="GROQ_API_KEY"),
            ProviderConfig("nvidia", "k2", second.base_url, "m2", key_env="NVIDIA_API_KEY"),
        ),
        max_retries=kw.get("max_retries", 2),
        retry_window=0.3,
    )


NOTES_USER = "<document_excerpt>\n[[Page 1]]\nA sentence long enough to be treated as a point here.\n</document_excerpt>"


@pytest.mark.parametrize("status", [429, 413, 401, 402, 500])
def test_failover_to_second_provider(status):
    bad, good = FakeLLMServer(fail_first=99, fail_status=status).start(), FakeLLMServer().start()
    try:
        client = LLMClient(two_providers(bad, good))
        assert "points" in client.complete_json("TASK: CHUNK_NOTES", NOTES_USER)
        assert bad.request_count == 1 and good.request_count == 1        # switched immediately, no waiting
        assert client.models_used == ["nvidia:m2"]
        # second call: a rate-limited/disabled provider is not hammered again
        client.complete_json("TASK: CHUNK_NOTES", NOTES_USER)
        assert bad.request_count == (2 if status == 413 else 1)
    finally:
        bad.shutdown(); good.shutdown()


def test_all_providers_rate_limited_raises_friendly_error():
    a, b = FakeLLMServer(fail_first=99).start(), FakeLLMServer(fail_first=99).start()
    try:
        with pytest.raises(LLMRateLimitError) as ei:
            LLMClient(two_providers(a, b, max_retries=1)).complete_json("TASK: CHUNK_NOTES", NOTES_USER)
        assert "OPENROUTER_API_KEY" in ei.value.user_message
    finally:
        a.shutdown(); b.shutdown()


def test_bad_key_message_names_the_right_variable():
    bad, good = FakeLLMServer(fail_first=99, fail_status=401).start(), FakeLLMServer(fail_first=99, fail_status=401).start()
    try:
        with pytest.raises(LLMConfigError) as ei:
            LLMClient(two_providers(bad, good)).complete_json("TASK: CHUNK_NOTES", NOTES_USER)
        assert "NVIDIA_API_KEY" in ei.value.user_message or "GROQ_API_KEY" in ei.value.user_message
    finally:
        bad.shutdown(); good.shutdown()


def test_parallel_calls_are_spread_across_providers():
    from concurrent.futures import ThreadPoolExecutor

    a, b = FakeLLMServer().start(), FakeLLMServer().start()
    try:
        client = LLMClient(two_providers(a, b))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: client.complete_json("TASK: CHUNK_NOTES", NOTES_USER), range(8)))
        assert a.request_count >= 2 and b.request_count >= 2        # not all on the first provider
        assert sorted(client.models_used) == ["groq:m1", "nvidia:m2"]
        assert all(s.in_flight == 0 for s in client._states)        # reservations released
    finally:
        a.shutdown(); b.shutdown()


def test_think_blocks_are_ignored():
    assert extract_json('<think>maybe {"x": 1}?</think>\n{"a": 2}') == {"a": 2}
