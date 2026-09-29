"""Runtime configuration, read from environment variables (optionally a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader (KEY=VALUE lines). Real environment variables always win."""
    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    provider: str = "openai"        # "openai" (any OpenAI-compatible API) or "anthropic"
    timeout: float = 120.0
    max_retries: int = 5
    max_output_tokens: int = 6000
    concurrency: int = 3
    json_mode: bool = True

    @classmethod
    def missing_vars(cls) -> list[str]:
        return [v for v in ("LLM_API_KEY", "LLM_MODEL") if not os.environ.get(v, "").strip()]

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Raises LLMConfigError with an actionable message if required variables are missing."""
        from .errors import LLMConfigError

        missing = cls.missing_vars()
        if missing:
            raise LLMConfigError(
                "The AI service is not configured. Set the environment variable(s) "
                + ", ".join(missing)
                + " (and optionally LLM_BASE_URL) and restart the app. See README.md > Environment Variables.",
                detail="missing env: " + ",".join(missing),
            )
        provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower() or "openai"
        default_url = "https://api.anthropic.com" if provider == "anthropic" else "https://api.openai.com/v1"
        return cls(
            api_key=os.environ["LLM_API_KEY"].strip(),
            base_url=(os.environ.get("LLM_BASE_URL", "").strip() or default_url).rstrip("/"),
            model=os.environ["LLM_MODEL"].strip(),
            provider=provider,
            timeout=_float("LLM_TIMEOUT", 120.0),
            max_retries=_int("LLM_MAX_RETRIES", 5),
            max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 6000),
            concurrency=max(1, _int("LLM_CONCURRENCY", 3)),
            json_mode=os.environ.get("LLM_JSON_MODE", "true").strip().lower() not in ("0", "false", "no"),
        )


@dataclass(frozen=True)
class Limits:
    max_pdf_mb: int = 25
    max_pages: int = 300
    chunk_chars: int = 12000          # target size of one map-stage chunk
    single_pass_chars: int = 40000    # documents up to this size skip map-reduce
    notes_budget_chars: int = 30000   # max size of the notes handed to the final synthesis

    @classmethod
    def from_env(cls) -> "Limits":
        d = cls()
        return cls(
            max_pdf_mb=_int("MAX_PDF_MB", d.max_pdf_mb),
            max_pages=_int("MAX_PDF_PAGES", d.max_pages),
            chunk_chars=_int("CHUNK_CHARS", d.chunk_chars),
            single_pass_chars=_int("SINGLE_PASS_CHARS", d.single_pass_chars),
            notes_budget_chars=_int("NOTES_BUDGET_CHARS", d.notes_budget_chars),
        )
