"""LLM client with multi-provider failover, built on httpx.

Providers (Groq, OpenRouter, NVIDIA, OpenAI, Anthropic or any OpenAI-compatible endpoint) are
configured through environment variables (see utils/config.py) and tried in order:

- 429 / rate limit  -> that provider cools down for the time it asks for; the call moves on to
                       the next provider immediately (no waiting while another one is free)
- 413 / too large   -> skip that provider for this call (e.g. a small tokens-per-minute tier)
- 5xx / network     -> short cool-down, try the next provider
- 401/403/402/404   -> the provider is disabled for this client (bad key, no credit, bad model)

Every failure surfaces as an LLMError subclass with a user-safe message.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field

import httpx

from ..utils.config import LLMConfig, ProviderConfig
from ..utils.errors import (
    CancelledError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

log = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 65.0
MIN_RATE_LIMIT_COOLDOWN = 2.0
MAX_OUTPUT_CAP = 12000            # ceiling when retrying a cut-off answer with a bigger budget
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences, <think> blocks and prose."""
    text = _THINK.sub("", text).strip()
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


@dataclass
class _ProviderState:
    cfg: ProviderConfig
    json_mode: bool = True
    cooldown_until: float = 0.0
    disabled: LLMError | None = None
    calls: int = 0
    in_flight: int = 0


class _RateLimited(Exception):
    def __init__(self, error: LLMError, delay: float):
        self.error, self.delay = error, delay


class _Transient(Exception):
    def __init__(self, error: LLMError, delay: float = 3.0):
        self.error, self.delay = error, delay


class _TooLarge(Exception):
    def __init__(self, error: LLMError, truncated: bool = False):
        self.error, self.truncated = error, truncated


class _Retry(Exception):
    """Outcome of a failed attempt: how much it costs and whether to avoid that provider."""

    def __init__(self, error: LLMError, cost: int = 0, skip: bool = False, failed: bool = False, truncated: bool = False):
        self.error, self.cost, self.skip, self.failed, self.truncated = error, cost, skip, failed, truncated


class _GiveUp(Exception):
    pass


@dataclass
class _CallState:
    skipped: set[str] = field(default_factory=set)   # unusable for this call (request too large)
    failed: set[str] = field(default_factory=set)    # already failed during this call


class ProviderPool:
    """Provider health (cool-downs, disabled keys) and the HTTP connection pool.

    Share one pool between all clients in a server process so a rate limit seen by one job is
    respected by the next.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.states = [_ProviderState(p, json_mode=config.json_mode) for p in config.providers]
        self.lock = threading.Lock()
        self.http = httpx.Client(timeout=httpx.Timeout(config.timeout, connect=15.0))

    def close(self) -> None:
        self.http.close()


class LLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        pool: ProviderPool | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.config = pool.config if pool else (config or LLMConfig.from_env())
        self._own_pool = pool is None
        self._pool = pool or ProviderPool(self.config)
        self._states = self._pool.states
        self._lock = self._pool.lock
        self._http = self._pool.http
        self._used: dict[str, int] = {}
        self._cancel = cancel_event or threading.Event()
        self.calls = 0

    @property
    def models_used(self) -> list[str]:
        """Provider:model pairs that answered calls made through THIS client."""
        return [f"{s.cfg.name}:{s.cfg.model}" for s in self._states if self._used.get(s.cfg.name)]

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise CancelledError()

    def _sleep(self, seconds: float) -> None:
        """Sleep that wakes up immediately when the job is cancelled."""
        if self._cancel.wait(timeout=max(0.0, seconds)):
            raise CancelledError()

    # ------------------------------------------------------------------ public
    def complete(self, system: str, user: str, *, want_json: bool = False, max_tokens: int | None = None) -> str:
        self._check_cancel()
        with self._lock:
            self.calls += 1
        max_tokens = max_tokens or self.config.max_output_tokens
        usable = sum(s.disabled is None for s in self._states)
        if self.config.hedge_after <= 0 or usable < 2:
            return self._with_failover(system, user, want_json, max_tokens)
        return self._hedged(system, user, want_json, max_tokens)

    def _hedged(self, system: str, user: str, want_json: bool, max_tokens: int) -> str:
        """Run the call; if it has not answered after `hedge_after` seconds, send the same request
        to a second provider (load balancing picks a different one) and take whichever answers
        first. Removes the long tail when one provider is slow; costs tokens only for slow calls."""
        results: queue.Queue = queue.Queue()

        def run() -> None:
            try:
                results.put((True, self._with_failover(system, user, want_json, max_tokens)))
            except BaseException as exc:      # noqa: BLE001 - forwarded to the caller
                results.put((False, exc))

        threading.Thread(target=run, daemon=True, name="llm-call").start()
        started, pending, first_error = 1, 1, None
        deadline_hedge = time.monotonic() + self.config.hedge_after
        while True:
            timeout = None if started > 1 else max(0.0, deadline_hedge - time.monotonic())
            try:
                ok, value = results.get(timeout=timeout)
            except queue.Empty:
                self._check_cancel()
                log.info("call slower than %.0fs; hedging on another provider", self.config.hedge_after)
                threading.Thread(target=run, daemon=True, name="llm-hedge").start()
                started += 1
                pending += 1
                continue
            pending -= 1
            if ok:
                return value
            if isinstance(value, CancelledError):
                raise value
            first_error = first_error or value
            if pending == 0:
                raise first_error

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

    # ---------------------------------------------------------------- failover
    def _pick(self, call: _CallState) -> tuple[_ProviderState | None, float]:
        """Provider for the next attempt and how long to wait for it (0 = ready now).

        Load-balances parallel calls: among ready providers it takes the one with the fewest
        requests in flight (ties -> configured order), so concurrent chunks are spread over
        different providers/models instead of all queuing on the first one. The chosen
        provider's in-flight count is reserved here and released by the caller.
        """
        with self._lock:
            usable = [s for s in self._states if s.disabled is None and s.cfg.name not in call.skipped]
            if not usable:
                return None, 0.0
            now = time.monotonic()
            ready = [s for s in usable if s.cooldown_until <= now]
            fresh = [s for s in ready if s.cfg.name not in call.failed]
            pool = fresh or ready
            if pool:
                order = {id(s): i for i, s in enumerate(self._states)}
                chosen = min(pool, key=lambda s: (s.in_flight, order[id(s)]))
                chosen.in_flight += 1
                return chosen, 0.0
            soonest = min(usable, key=lambda s: s.cooldown_until)
            soonest.in_flight += 1
            return soonest, soonest.cooldown_until - now

    def _with_failover(self, system: str, user: str, want_json: bool, max_tokens: int) -> str:
        """Rate limits are retried until `retry_window` seconds have passed (they clear on their
        own); other failures (5xx, timeouts, network) count against max_retries + #providers."""
        call = _CallState()
        last: LLMError | None = None
        failures_left = self.config.max_retries + len(self._states)
        deadline = time.monotonic() + self.config.retry_window
        while failures_left > 0:
            state, wait = self._pick(call)
            if state is None:
                break
            try:
                return self._attempt(state, wait, deadline, last, system, user, want_json, max_tokens)
            except _GiveUp:
                break
            except _Retry as r:
                last = r.error
                failures_left -= r.cost
                if r.truncated and max_tokens < MAX_OUTPUT_CAP:
                    # Reasoning models spend part of the budget thinking: give them more room.
                    max_tokens = min(max_tokens * 2, MAX_OUTPUT_CAP)
                    log.info("%s: output cut off; retrying with max_tokens=%d", state.cfg.name, max_tokens)
                    continue
                if r.skip:
                    call.skipped.add(state.cfg.name)
                if r.failed:
                    call.failed.add(state.cfg.name)
            finally:
                with self._lock:
                    state.in_flight -= 1
        if last is None:
            last = LLMConfigError("No AI provider is available (all were disabled).")
        raise last

    def _attempt(self, state: _ProviderState, wait: float, deadline: float, last: LLMError | None,
                 system: str, user: str, want_json: bool, max_tokens: int) -> str:
        """One attempt on one provider. Raises _Retry (try again) or _GiveUp (stop now)."""
        if time.monotonic() + wait > deadline and isinstance(last, LLMRateLimitError):
            raise _GiveUp()
        self._check_cancel()
        if wait > 0:
            log.info("All providers cooling down; waiting %.1fs for %s", wait, state.cfg.name)
            self._sleep(min(wait, MAX_WAIT_SECONDS))
        name = state.cfg.name
        try:
            text = self._request(state, system, user, want_json, max_tokens)
        except _RateLimited as exc:
            self._cool(state, max(exc.delay, MIN_RATE_LIMIT_COOLDOWN), "rate limited")
            raise _Retry(exc.error, failed=True)
        except _Transient as exc:
            self._cool(state, exc.delay, "temporary error")
            raise _Retry(exc.error, cost=1, failed=True)
        except _TooLarge as exc:
            if exc.truncated:
                raise _Retry(exc.error, cost=1, truncated=True)
            log.info("%s: %s; trying the next provider", name, exc.error.detail[:80] or "request too large")
            raise _Retry(exc.error, skip=True)
        except LLMConfigError as exc:
            with self._lock:
                state.disabled = exc
            log.warning("%s disabled: %s | %s", name, exc.user_message, exc.detail[:200])
            if len(self._states) == 1:
                raise
            raise _Retry(exc)
        except httpx.TimeoutException as exc:
            self._cool(state, 5.0, "timeout")
            raise _Retry(LLMTimeoutError(detail=f"{name}: {exc!r}"), cost=1, failed=True)
        except httpx.TransportError as exc:
            self._cool(state, 2.0, "network error")
            raise _Retry(LLMError(
                "Could not reach the AI service. Check the provider base URL and the network connection.",
                detail=f"{name}: {exc!r}"), cost=1, failed=True)
        with self._lock:
            state.calls += 1
            self._used[name] = self._used.get(name, 0) + 1
        return text

    def _cool(self, state: _ProviderState, delay: float, why: str) -> None:
        with self._lock:
            state.cooldown_until = time.monotonic() + delay
        log.info("%s %s; cooling down %.1fs", state.cfg.name, why, delay)

    # ---------------------------------------------------------------- one HTTP call
    def _request(self, state: _ProviderState, system: str, user: str, want_json: bool, max_tokens: int) -> str:
        p = state.cfg
        if p.protocol == "anthropic":
            url = f"{p.base_url}/v1/messages"
            headers = {"x-api-key": p.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            body = {
                "model": p.model,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        else:
            url = f"{p.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {p.api_key}", "content-type": "application/json"}
            if p.name == "openrouter":
                headers["X-Title"] = "ResearchOS"
            body = {
                "model": p.model,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if want_json and state.json_mode:
                body["response_format"] = {"type": "json_object"}

        resp = self._http.post(url, headers=headers, json=body)

        if resp.status_code in (400, 422) and "response_format" in body:
            # Some OpenAI-compatible servers/models reject response_format; remember and retry without it.
            state.json_mode = False
            body.pop("response_format")
            resp = self._http.post(url, headers=headers, json=body)

        status, text = resp.status_code, resp.text[:400]
        if status in (401, 403):
            raise LLMConfigError(
                f"The AI provider '{p.name}' rejected the API key. Check {p.key_env}.", detail=text
            )
        if status == 402:
            raise LLMConfigError(f"The AI provider '{p.name}' has no credit left for this key.", detail=text)
        if status == 404:
            raise LLMConfigError(
                f"The AI provider '{p.name}' does not offer model '{p.model}' (or the base URL is wrong).",
                detail=text,
            )
        if status == 413 or (status == 400 and "too large" in text.lower()):
            raise _TooLarge(
                LLMError(
                    f"The request is too large for the '{p.name}' plan. Lower CHUNK_CHARS / SINGLE_PASS_CHARS "
                    "/ NOTES_BUDGET_CHARS, or add another provider.",
                    detail=text,
                )
            )
        if status == 429:
            raise _RateLimited(
                LLMRateLimitError(
                    "The AI providers are rate limiting requests. Wait a minute and retry, or add another "
                    "provider (GROQ_API_KEY, OPENROUTER_API_KEY, NVIDIA_API_KEY).",
                    detail=f"{p.name}: {text}",
                ),
                _retry_delay(resp),
            )
        if status >= 500:
            raise _Transient(LLMError("The AI service had a temporary error.", detail=f"{p.name} {status} {text}"))
        if status >= 400:
            raise LLMError(f"The AI provider '{p.name}' rejected the request (HTTP {status}).", detail=text)

        try:
            payload = resp.json()
            if p.protocol == "anthropic":
                content = "".join(b.get("text", "") for b in payload["content"] if b.get("type") == "text")
                truncated = payload.get("stop_reason") == "max_tokens"
            else:
                choice = payload["choices"][0]
                content = choice["message"].get("content") or ""
                truncated = choice.get("finish_reason") == "length"
        except (ValueError, KeyError, IndexError, TypeError):
            raise _Transient(LLMResponseError(detail=f"{p.name}: unexpected payload {text}"), 1.0)
        if truncated and want_json:
            # Cut-off JSON cannot be repaired; another model may be less verbose.
            raise _TooLarge(LLMResponseError(
                "The AI response was cut off before it was complete. Try again, or raise LLM_MAX_OUTPUT_TOKENS.",
                detail=f"output truncated at {max_tokens} tokens"), truncated=True)
        if not content.strip():
            raise _Transient(LLMResponseError("The AI service returned an empty response.", detail=p.name), 1.0)
        return content

    def close(self) -> None:
        if self._own_pool:
            self._pool.close()


def _retry_delay(resp: httpx.Response) -> float:
    """Seconds to wait before this provider is usable again (Retry-After or the provider's hint)."""
    retry_after = resp.headers.get("retry-after", "")
    if retry_after.replace(".", "", 1).isdigit():
        return min(float(retry_after), MAX_WAIT_SECONDS)
    # Groq style: "try again in 23.49s", "try again in 1m5.2s", "try again in 520ms"
    hint = re.search(r"try again in (?:(\d+)m(?!s))?\s*([\d.]+)(ms|s)", resp.text)
    if hint:
        seconds = float(hint.group(2)) / (1000 if hint.group(3) == "ms" else 1)
        return min(float(hint.group(1) or 0) * 60 + seconds + 0.5, MAX_WAIT_SECONDS)
    return 10.0
