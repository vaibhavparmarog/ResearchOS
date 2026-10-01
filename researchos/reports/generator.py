"""End-to-end orchestration: PDF bytes -> validated Report.

PDF -> extract -> clean -> page-aware chunks -> (map: notes per chunk -> reduce) -> final report
    -> validate against the PDF -> Report
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable

from ..analysis.analyzer import analyze_chunks
from ..analysis.synthesizer import reduce_notes, write_final_report
from ..analysis.validator import validate_report
from ..llm.client import LLMClient
from ..pdf.chunker import analysis_chars, chunk_document
from ..pdf.extractor import ExtractedDocument, extract_pdf
from ..utils.config import Limits
from .models import Report

log = logging.getLogger(__name__)

# on_progress(fraction 0..1, human readable message)
Progress = Callable[[float, str], None]


def extract_document(
    data: bytes | None, filename: str, limits: Limits | None = None, on_progress: Progress | None = None
) -> ExtractedDocument:
    """Stage 1 only (no LLM). Lets the UI show file facts before analysis starts. OCR reports progress."""
    return extract_pdf(data, filename, limits, on_progress=on_progress)


def generate_report(
    doc: ExtractedDocument,
    client: LLMClient,
    limits: Limits | None = None,
    on_progress: Progress | None = None,
) -> Report:
    """Analyse an already-extracted document. Everything sent to the LLM comes from `doc`."""
    limits = limits or Limits.from_env()
    say = on_progress or (lambda f, m: None)
    started = time.time()
    calls_before = client.calls
    warnings: list[str] = []

    say(0.05, "Splitting the document into page-aware chunks")
    chunks = chunk_document(doc, limits)
    single_pass = analysis_chars(doc) <= limits.single_pass_chars or len(chunks) <= 1
    notes = None

    if not single_pass:
        say(0.10, f"Analysing {len(chunks)} sections of the document")
        notes, chunk_warnings = analyze_chunks(
            client, doc, chunks, lambda f, m: say(0.10 + 0.55 * f, m)
        )
        warnings += chunk_warnings
        say(0.65, "Merging section notes")
        notes = reduce_notes(client, doc, notes, limits, lambda f, m: say(0.65 + 0.10 * f, m))

    say(0.78, "Writing the report" + (" from the full text" if single_pass else " from the merged notes"))
    raw = write_final_report(client, doc, limits, notes)

    say(0.92, "Verifying page references and quotes against the PDF")
    meta = {
        "model": ", ".join(getattr(client, "models_used", None) or [client.config.model]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": "single-pass" if single_pass else "map-reduce",
        "chunks": len(chunks),
        "llm_calls": client.calls - calls_before,
        "seconds": round(time.time() - started, 1),
        "sha256": doc.sha256,
    }
    report = validate_report(raw, doc, meta)
    report.warnings += warnings
    say(1.0, "Done")
    return report
