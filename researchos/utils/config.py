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


# name -> (default base URL, default model, wire protocol). All but "anthropic" speak the
# OpenAI chat-completions protocol. Models can be overridden with <NAME>_MODEL.
PROVIDER_PRESETS: dict[str, tuple[str, str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "openai"),
    "openrouter": ("https://openrouter.ai/api/v1", "openai/gpt-oss-120b", "openai"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "nvidia/nemotron-3-super-120b-a12b", "openai"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "openai"),
    "anthropic": ("https://api.anthropic.com", "", "anthropic"),
}
AUTO_ORDER = ("groq", "openrouter", "nvidia", "openai", "anthropic")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    protocol: str = "openai"        # "openai" (chat-completions) or "anthropic" (messages API)
    key_env: str = "LLM_API_KEY"    # where the key came from (for error messages)


def _providers_from_env(name: str) -> list[ProviderConfig]:
    """One entry per model: <NAME>_MODEL may list several ("a,b"), each with its own rate-limit bucket."""
    prefix = name.upper()
    key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    if not key:
        return []
    url, default_model, protocol = PROVIDER_PRESETS.get(name, ("", "", "openai"))
    models = [m.strip() for m in os.environ.get(f"{prefix}_MODEL", "").split(",") if m.strip()] or [default_model]
    url = (os.environ.get(f"{prefix}_BASE_URL", "").strip() or url).rstrip("/")
    if not url:
        return []
    return [
        ProviderConfig(name if i == 0 else f"{name}#{i + 1}", key, url, m, protocol, f"{prefix}_API_KEY")
        for i, m in enumerate(models) if m
    ]


def providers_from_env() -> list[ProviderConfig]:
    """Configured providers in failover order.

    - LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / LLM_PROVIDER: one generic provider (tried first).
    - GROQ_*, OPENROUTER_*, NVIDIA_*, OPENAI_*, ANTHROPIC_*: named providers (<NAME>_API_KEY,
      optional <NAME>_MODEL / <NAME>_BASE_URL).
    - LLM_PROVIDERS=groq,openrouter,nvidia sets the order (default: the order above).
    """
    out: list[ProviderConfig] = []
    generic_key, generic_model = os.environ.get("LLM_API_KEY", "").strip(), os.environ.get("LLM_MODEL", "").strip()
    if generic_key and generic_model:
        kind = os.environ.get("LLM_PROVIDER", "openai").strip().lower() or "openai"
        protocol = "anthropic" if kind == "anthropic" else "openai"
        default_url = PROVIDER_PRESETS.get(kind, PROVIDER_PRESETS["openai"])[0]
        url = (os.environ.get("LLM_BASE_URL", "").strip() or default_url).rstrip("/")
        out.append(ProviderConfig("custom" if kind not in PROVIDER_PRESETS else kind, generic_key, url, generic_model, protocol))
    order = [n.strip().lower() for n in os.environ.get("LLM_PROVIDERS", "").split(",") if n.strip()] or list(AUTO_ORDER)
    for name in order:
        for p in _providers_from_env(name):
            if not any(o.api_key == p.api_key and o.base_url == p.base_url and o.model == p.model for o in out):
                out.append(p)
    return out


@dataclass(frozen=True)
class LLMConfig:
    providers: tuple[ProviderConfig, ...]
    timeout: float = 120.0
    max_retries: int = 5
    max_output_tokens: int = 6000
    concurrency: int = 3
    json_mode: bool = True
    retry_window: float = 240.0     # seconds a single call may keep waiting out rate limits

    @property
    def model(self) -> str:
        return self.providers[0].model

    @classmethod
    def single(cls, api_key: str, base_url: str, model: str, protocol: str = "openai", **kw) -> "LLMConfig":
        return cls(providers=(ProviderConfig("custom", api_key, base_url.rstrip("/"), model, protocol),), **kw)

    @classmethod
    def missing_vars(cls) -> list[str]:
        if providers_from_env():
            return []
        return ["GROQ_API_KEY / OPENROUTER_API_KEY / NVIDIA_API_KEY (or LLM_API_KEY + LLM_MODEL)"]

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Raises LLMConfigError with an actionable message if no provider is configured."""
        from .errors import LLMConfigError

        providers = providers_from_env()
        if not providers:
            raise LLMConfigError(
                "The AI service is not configured. Set at least one of GROQ_API_KEY, OPENROUTER_API_KEY, "
                "NVIDIA_API_KEY (or LLM_API_KEY + LLM_MODEL) and restart the app. See README.md > Environment Variables.",
                detail="no provider configured",
            )
        return cls(
            providers=tuple(providers),
            timeout=_float("LLM_TIMEOUT", 120.0),
            max_retries=_int("LLM_MAX_RETRIES", 5),
            max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 6000),
            # default: one parallel request per provider/model slot (each has its own rate limit)
            concurrency=max(1, _int("LLM_CONCURRENCY", max(3, len(providers)))),
            json_mode=os.environ.get("LLM_JSON_MODE", "true").strip().lower() not in ("0", "false", "no"),
            retry_window=_float("LLM_RETRY_WINDOW", 240.0),
        )


@dataclass(frozen=True)
class Limits:
    max_pdf_mb: int = 25
    max_pages: int = 300
    chunk_chars: int = 12000          # target size of one map-stage chunk
    single_pass_chars: int = 40000    # documents up to this size skip map-reduce
    notes_budget_chars: int = 30000   # max size of the notes handed to the final synthesis
    max_ocr_pages: int = 60           # cap on pages OCR'd per document (OCR is ~1-2 s/page)
    skip_references: bool = True      # keep the bibliography out of the LLM input

    @classmethod
    def from_env(cls) -> "Limits":
        d = cls()
        return cls(
            max_pdf_mb=_int("MAX_PDF_MB", d.max_pdf_mb),
            max_pages=_int("MAX_PDF_PAGES", d.max_pages),
            chunk_chars=_int("CHUNK_CHARS", d.chunk_chars),
            single_pass_chars=_int("SINGLE_PASS_CHARS", d.single_pass_chars),
            notes_budget_chars=_int("NOTES_BUDGET_CHARS", d.notes_budget_chars),
            max_ocr_pages=_int("MAX_OCR_PAGES", d.max_ocr_pages),
            skip_references=os.environ.get("SKIP_REFERENCES", "true").strip().lower() not in ("0", "false", "no"),
        )
