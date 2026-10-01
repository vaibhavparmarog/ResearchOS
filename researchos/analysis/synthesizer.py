"""Reduce + final stages: merge chunk notes hierarchically, then write the final structured report."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..llm.client import LLMClient
from ..llm.prompts import (
    CONDENSE_SYSTEM,
    FINAL_SYSTEM,
    condense_user_prompt,
    final_user_prompt_from_notes,
    final_user_prompt_from_text,
)
from ..pdf.chunker import full_text, page_marker
from ..pdf.extractor import ExtractedDocument
from ..utils.config import Limits
from ..utils.errors import LLMResponseError
from .validator import PageIndex, sanitize_notes

StageProgress = Callable[[float, str], None]

MAX_REDUCE_LEVELS = 6
FINAL_MIN_OUTPUT_TOKENS = 8000


def _size(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False))


def fit_notes(note: dict, max_chars: int) -> dict:
    """Trim a note (dropping trailing items of its largest list) until its JSON is <= max_chars.

    Bounds every note relative to the budget, so hierarchical merging always converges no matter
    how verbose the model is.
    """
    note = dict(note)
    keys = ("points", "concepts", "data", "limitations_stated")
    while _size(note) > max_chars:
        candidates = [k for k in keys if note.get(k)]
        if not candidates:
            note["summary"] = str(note.get("summary", ""))[: max(200, max_chars // 4)]
            break
        biggest = max(candidates, key=lambda k: _size(note[k]))
        note[biggest] = note[biggest][:-1]
    return note


def _batches(notes: list[dict], budget: int) -> list[list[dict]]:
    """Consecutive groups of >=2 notes whose combined JSON size stays within `budget`."""
    groups: list[list[dict]] = []
    cur: list[dict] = []
    cur_size = 0
    for n in notes:
        s = _size(n)
        if len(cur) >= 2 and cur_size + s > budget:
            groups.append(cur)
            cur, cur_size = [], 0
        cur.append(n)
        cur_size += s
    if cur:
        if len(cur) == 1 and groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


def reduce_notes(
    client: LLMClient,
    doc: ExtractedDocument,
    notes: list[dict],
    limits: Limits,
    on_progress: StageProgress | None = None,
) -> list[dict]:
    """Hierarchically condense notes until they fit the synthesis budget."""
    idx = PageIndex(doc)
    level = 0
    while len(notes) > 1 and _size(notes) > limits.notes_budget_chars and level < MAX_REDUCE_LEVELS:
        level += 1
        notes = [fit_notes(n, limits.notes_budget_chars) for n in notes]
        groups = _batches(notes, limits.notes_budget_chars)
        if on_progress:
            on_progress(0.0, f"Condensing notes (level {level}, {len(groups)} group(s))")

        def merge(group: list[dict]) -> dict:
            if len(group) == 1:
                return group[0]
            pages = sorted({p for n in group for p in n.get("pages_covered", [])})
            raw = client.complete_json(
                CONDENSE_SYSTEM, condense_user_prompt(doc, group),
                max_tokens=min(client.config.max_output_tokens, 3000),
            )
            merged = sanitize_notes(raw, pages, idx)
            if not (merged["points"] or merged["concepts"] or merged["data"]):
                raise LLMResponseError(detail="condensed notes were empty")
            return fit_notes(merged, max(1500, limits.notes_budget_chars // 2))

        with ThreadPoolExecutor(max_workers=client.config.concurrency) as pool:
            notes = list(pool.map(merge, groups))
    return notes


def write_final_report(
    client: LLMClient,
    doc: ExtractedDocument,
    limits: Limits,
    notes: list[dict] | None,
) -> dict:
    """Final synthesis. `notes=None` means single-pass over the whole (short) document."""
    valid_pages = [p.number for p in doc.analysis_pages]
    if notes is None:
        user = final_user_prompt_from_text(doc, full_text(doc), valid_pages)
    else:
        # title / authors / date live at the top of page 1; keep this small for low-TPM providers
        opening = "\n\n".join(f"{page_marker(p.number)}\n{p.text}" for p in doc.text_pages[:2])[:4000]
        user = final_user_prompt_from_notes(doc, {"parts": notes} if len(notes) > 1 else notes[0], opening)
    # The final report is long and reasoning models also spend tokens thinking: start with enough
    # room (a cut-off answer costs a whole extra call). Small-quota providers skip it instantly (413).
    return client.complete_json(FINAL_SYSTEM, user, max_tokens=max(client.config.max_output_tokens, FINAL_MIN_OUTPUT_TOKENS))
