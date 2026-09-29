"""Map stage: turn each page-aware chunk of the uploaded document into verified structured notes."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from ..llm.client import LLMClient
from ..llm.prompts import CHUNK_SYSTEM, chunk_user_prompt
from ..pdf.chunker import Chunk
from ..pdf.extractor import ExtractedDocument
from ..utils.errors import LLMResponseError
from .validator import PageIndex, sanitize_notes

log = logging.getLogger(__name__)

# progress(fraction_within_stage, message)
StageProgress = Callable[[float, str], None]


def analyze_chunks(
    client: LLMClient,
    doc: ExtractedDocument,
    chunks: list[Chunk],
    on_progress: StageProgress | None = None,
) -> tuple[list[dict], list[str]]:
    """Return (notes in document order, warnings). Chunks whose output is unusable are skipped."""
    idx = PageIndex(doc)
    results: dict[int, dict] = {}
    warnings: list[str] = []

    def work(chunk: Chunk) -> dict:
        raw = client.complete_json(
            CHUNK_SYSTEM, chunk_user_prompt(doc, chunk, len(chunks)),
            max_tokens=min(client.config.max_output_tokens, 3500),
        )
        return sanitize_notes(raw, chunk.pages, idx)

    with ThreadPoolExecutor(max_workers=client.config.concurrency) as pool:
        futures = {pool.submit(work, c): c for c in chunks}
        done = 0
        for fut in as_completed(futures):
            chunk = futures[fut]
            try:
                results[chunk.index] = fut.result()
            except LLMResponseError as exc:
                log.warning("Chunk %d unusable: %s", chunk.index, exc.detail)
                warnings.append(f"The AI response for {chunk.label} could not be parsed; that part is not fully covered.")
            done += 1
            if on_progress:
                on_progress(done / len(chunks), f"Analysed {done} of {len(chunks)} sections of the document")

    if not results:
        raise LLMResponseError("The AI service did not return usable analysis for any part of the document.")
    return [results[i] for i in sorted(results)], warnings
