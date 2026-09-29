"""Thin LLM client (OpenAI-compatible chat API, or Anthropic Messages API) built on httpx.

Configured only through environment variables (see utils/config.py). Handles timeouts,
rate limits, transient 5xx errors, and malformed JSON, and maps every failure onto an
LLMError subclass with a user-safe message.
"""

from __future__ import annotations

import json
import logging
import re
import time

import httpx

from ..utils.config import LLMConfig
from ..utils.errors import (
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

log = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("expected a JSON object", text, 0)
    return value


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self._json_mode = self.config.json_mode
        self._http = httpx.Client(timeout=httpx.Timeout(self.config.timeout, connect=15.0))
        self.calls = 0

    # ------------------------------------------------------------------ public
    def complete(self, system: str, user: str, *, want_json: bool = False, max_tokens: int | None = None) -> str:
        self.calls += 1
        return self._with_retries(system, user, want_json, max_tokens or self.config.max_output_tokens)

    def complete_json(self, system: str, user: str, *, max_tokens: int | None = None) -> dict:
        raw = self.complete(system, user, want_json=True, max_tokens=max_tokens)
        try:
            return extract_json(raw)
        except json.JSONDecodeError:
            log.warning("Malformed JSON from model; asking it to repair (%d chars)", len(raw))
        repair = self.complete(
            "You repair malformed JSON. Output ONLY the corrected JSON object, nothing else. "
            "Do not add, remove, or change any content.",
            raw[:60000],
            want_json=True,
            max_tokens=max_tokens,
        )
        try:
            return extract_json(repair)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(detail=f"unparseable JSON after repair: {repair[:300]!r}") from exc

    # ---------------------------------------------------------------- internals
    def _with_retries(self, system: str, user: str, want_json: bool, max_tokens: int) -> str:
        cfg = self.config
        last: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                return self._request(system, user, want_json, max_tokens)
            except httpx.TimeoutException as exc:
                last = LLMTimeoutError(detail=repr(exc))
                delay = 2.0 * (attempt + 1)
            except httpx.TransportError as exc:
                last = LLMError(
                    "Could not reach the AI service. Check LLM_BASE_URL and your network connection.",
                    detail=repr(exc),
                )
                delay = 2.0 * (attempt + 1)
            except _Retryable as exc:
                last, delay = exc.error, exc.delay
            except LLMError:
                raise
            if attempt < cfg.max_retries:
                log.info("LLM call failed (%s); retrying in %.1fs", type(last).__name__, delay)
                time.sleep(delay)
        assert last is not None
        raise last

    def _request(self, system: str, user: str, want_json: bool, max_tokens: int) -> str:
        cfg = self.config
        if cfg.provider == "anthropic":
            url = f"{cfg.base_url}/v1/messages"
            headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            body = {
                "model": cfg.model,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        else:
            url = f"{cfg.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {cfg.api_key}", "content-type": "application/json"}
            body = {
                "model": cfg.model,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if want_json and self._json_mode:
                body["response_format"] = {"type": "json_object"}

        resp = self._http.post(url, headers=headers, json=body)

        if resp.status_code == 400 and "response_format" in body:
            # Some OpenAI-compatible servers reject response_format; remember and retry without it.
            self._json_mode = False
            body.pop("response_format")
            resp = self._http.post(url, headers=headers, json=body)

        if resp.status_code in (401, 403):
            raise LLMConfigError(
                "The AI service rejected the API key (authentication failed). Check LLM_API_KEY.",
                detail=resp.text[:300],
            )
        if resp.status_code == 404:
            raise LLMConfigError(
                "The AI service endpoint or model was not found. Check LLM_BASE_URL and LLM_MODEL.",
                detail=resp.text[:300],
            )
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after", "")
            hint = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", resp.text)      # e.g. Groq: "try again in 23.49s"
            if retry_after.replace(".", "", 1).isdigit():
                delay = float(retry_after)
            elif hint:
                delay = float(hint.group(1) or 0) * 60 + float(hint.group(2)) + 1.0
            else:
                delay = 5.0
            raise _Retryable(
                LLMRateLimitError(
                    "The AI service is rate limiting requests (likely a tokens-per-minute cap on your plan). "
                    "Wait a minute and retry. If it keeps happening, set LLM_CONCURRENCY=1 and lower "
                    "CHUNK_CHARS / SINGLE_PASS_CHARS / NOTES_BUDGET_CHARS so each request is smaller.",
                    detail=resp.text[:300],
                ),
                min(delay, 65.0),
            )
        if resp.status_code >= 500:
            raise _Retryable(LLMError("The AI service had a temporary error.", detail=f"{resp.status_code} {resp.text[:300]}"), 3.0)
        if resp.status_code >= 400:
            raise LLMError(
                f"The AI service rejected the request (HTTP {resp.status_code}).",
                detail=resp.text[:500],
            )

        try:
            payload = resp.json()
            if cfg.provider == "anthropic":
                text = "".join(b.get("text", "") for b in payload["content"] if b.get("type") == "text")
            else:
                text = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(detail=f"unexpected payload: {resp.text[:300]}") from exc
        if not text or not text.strip():
            raise LLMResponseError("The AI service returned an empty response.")
        return text

    def close(self) -> None:
        self._http.close()


class _Retryable(Exception):
    def __init__(self, error: LLMError, delay: float):
        self.error, self.delay = error, delay
