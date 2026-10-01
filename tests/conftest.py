"""Shared fixtures: two unrelated, generated PDFs (A and B) plus a long one."""

from __future__ import annotations

import os

import pytest

from researchos.utils.config import Limits

from .pdf_factory import make_pdf

# Two deliberately unrelated documents with disjoint vocabulary.
PDF_A_PAGES = [
    "Coral Bleaching Resilience in the Outer Reef\nMaren Solvik and Tadeo Quillfeather\n2019\n\n"
    "This study examines how zooxanthellae density influences thermal tolerance in staghorn coral colonies.",
    "Methods. Researchers transplanted 240 fragments across six reef stations and recorded seawater temperature "
    "every ten minutes. Fragment survival was scored weekly by divers.",
    "Results. Survival at the shaded stations reached 87.5% while exposed stations recorded 41.2% survival. "
    "Thermal tolerance correlated strongly with symbiont density in every trial.",
    "Limitations. The authors state that the experiment covered a single summer season and cannot separate "
    "the effect of shading from that of water flow.",
]
PDF_B_PAGES = [
    "Quarterly Warehouse Logistics Review\nPrepared by Ingrid Haldorsen\n2023\n\n"
    "This report summarises forklift utilisation and pallet throughput across the Rotterdam distribution centre.",
    "Operations. Inbound pallets averaged 1,860 per day, and dock scheduling was moved to a two-shift rota "
    "to reduce truck waiting time.",
    "Financials. Operating cost per pallet fell to $3.42 in the third quarter, a decrease of 12% compared to "
    "the second quarter, driven by lower overtime.",
]


@pytest.fixture(scope="session")
def pdf_a() -> bytes:
    return make_pdf(PDF_A_PAGES)


@pytest.fixture(scope="session")
def pdf_b() -> bytes:
    return make_pdf(PDF_B_PAGES)


@pytest.fixture()
def limits() -> Limits:
    return Limits(max_pdf_mb=5, max_pages=100, chunk_chars=1500, single_pass_chars=3000, notes_budget_chars=6000)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_PROVIDER", "LLM_PROVIDERS"):
        monkeypatch.delenv(var, raising=False)
    for name in ("GROQ", "OPENROUTER", "NVIDIA", "OPENAI", "ANTHROPIC"):
        for suffix in ("_API_KEY", "_MODEL", "_BASE_URL"):
            monkeypatch.delenv(name + suffix, raising=False)
    monkeypatch.setenv("OCR_MODE", "off")      # tests must not depend on a local Tesseract install
    monkeypatch.setattr("researchos.llm.client.time.sleep", lambda s: None)   # no real retry back-off in tests
