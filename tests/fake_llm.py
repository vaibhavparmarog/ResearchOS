"""Deterministic stand-in for an LLM, used by tests and for offline UI verification.

IT IS NOT A MODEL. It builds its JSON answers purely from the text present in the prompt it is
given (page markers, sentences, numbers), which lets the tests prove that the pipeline really
passes the uploaded document's content to the LLM and that the validator works. It contains no
knowledge of any particular document.

`hallucinate=True` makes it additionally emit fabricated pages/quotes/authors so the validator
can be tested.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

PAGE_SPLIT = re.compile(r"\[\[Page (\d+)\]\]\n")


def _pages_in(block: str) -> dict[int, str]:
    parts = PAGE_SPLIT.split(block)
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _tag(user: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>\n?(.*?)\n?</{tag}>", user, re.S)
    return m.group(1) if m else ""


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.strip()) >= 40]


def _quote(sentence: str, n: int = 90) -> str:
    q = sentence[:n]
    return q[: q.rfind(" ")] if len(sentence) > n and " " in q else q


def _numbers(text: str) -> list[tuple[str, str]]:
    flat = re.sub(r"\s+", " ", text)
    out = []
    for m in re.finditer(r"([A-Za-z][A-Za-z \-]{3,40}?)\s(?:of|was|is|at|reached|to|:)?\s?(\$?\d[\d,]*(?:\.\d+)?\s?%?)", flat):
        out.append((m.group(1).strip(), m.group(2).strip()))
    return out[:4]


def _notes_for(pages: dict[int, str]) -> dict:
    points, data, concepts = [], [], []
    for pno, text in pages.items():
        sents = _sentences(text)
        for s in sents[:2]:
            points.append({"text": s, "kind": "claim", "pages": [pno], "quote": _quote(s)})
        for label, value in _numbers(text):
            data.append({"label": label, "value": value, "context": "", "pages": [pno]})
        caps = re.findall(r"\b[A-Z][a-z]{5,}\b", text)
        for term in sorted(set(caps), key=lambda t: -caps.count(t))[:1]:
            concepts.append({"term": term, "explanation": (sents[0] if sents else term)[:160], "pages": [pno]})
    first = next(iter(pages.values()), "")
    return {
        "summary": " ".join(_sentences(first)[:2]),
        "topics": [],
        "points": points,
        "concepts": concepts,
        "data": data,
        "limitations_stated": [],
    }


def fake_respond(system: str, user: str, hallucinate: bool = False) -> str:
    task = re.match(r"TASK: (\w+)", system)
    task = task.group(1) if task else "UNKNOWN"

    if task == "CHUNK_NOTES":
        return json.dumps(_notes_for(_pages_in(_tag(user, "document_excerpt"))))

    if task == "CONDENSE_NOTES":
        parts = [json.loads(m) for m in re.findall(r"<notes part=\d+>\n(.*?)\n</notes>", user, re.S)]
        merged = {"summary": " ".join(p.get("summary", "") for p in parts)[:600], "topics": [],
                  "points": [], "concepts": [], "data": [], "limitations_stated": []}
        for p in parts:
            for k in ("points", "concepts", "data", "limitations_stated"):
                merged[k] += p.get(k, [])[: 8 if k == "points" else 4]
        return json.dumps(merged)

    if task == "FINAL_REPORT":
        opening = _pages_in(_tag(user, "document_opening") or _tag(user, "document"))
        first_page = opening.get(min(opening), "") if opening else ""
        first_lines = [l.strip() for l in first_page.split("\n") if l.strip()]
        title = first_lines[0] if first_lines else None
        notes_json = _tag(user, "document_notes")
        if notes_json:
            n = json.loads(notes_json)
            parts = n["parts"] if "parts" in n else [n]
            points = [p for part in parts for p in part["points"]]
            data = [d for part in parts for d in part["data"]]
            concepts = [c for part in parts for c in part["concepts"]]
        else:
            points = _notes_for(opening)["points"]
            data = _notes_for(opening)["data"]
            concepts = _notes_for(opening)["concepts"]
        pts = points[:10]
        sections = []
        if pts:
            all_pages = sorted({p for pt in pts for p in pt["pages"]})
            sections.append({
                "key": "executive_summary", "heading": "Executive Summary", "basis": "document",
                "body_markdown": " ".join(pt["text"] for pt in pts[:3]),
                "pages": all_pages[:3], "evidence": [{"page": pt["pages"][0], "quote": pt["quote"]} for pt in pts[:2] if pt.get("quote")],
            })
            sections.append({
                "key": "key_findings", "heading": "Key Findings", "basis": "document",
                "body_markdown": "\n".join(f"- {pt['text']} (p. {pt['pages'][0]})" for pt in pts),
                "pages": all_pages, "evidence": [],
            })
        if concepts:
            sections.append({"key": "key_concepts", "heading": "Key Concepts", "basis": "document",
                             "body_markdown": "\n".join(f"- **{c['term']}**: {c['explanation']}" for c in concepts[:6]),
                             "pages": sorted({p for c in concepts[:6] for p in c["pages"]}), "evidence": []})
        sections.append({"key": "limitations_inferred", "heading": "Inferred Limitations", "basis": "interpretation",
                         "body_markdown": "AI interpretation: coverage limited to the text supplied.", "pages": [], "evidence": []})
        authors: list[str] = []
        if hallucinate:
            sections.append({
                "key": "results_evidence", "heading": "Results", "basis": "document",
                "body_markdown": "The study reports remarkable gains (see page 999) and further detail on page 1000–1002.",
                "pages": [999, 1000], "evidence": [{"page": 999, "quote": "This sentence was invented by the model and is nowhere in the file."}],
            })
            sections.append({"key": "methodology", "heading": "Methodology", "basis": "document",
                             "body_markdown": "An invented methodology with no page support.", "pages": [], "evidence": []})
            # wrong page declared for a real quote -> should be corrected
            if len(pts) >= 3:
                sections[0]["evidence"].append({"page": 998, "quote": pts[-1]["quote"]})
            authors = ["Dr. Nonexistent Person"]
            data = data + [{"label": "Invented accuracy", "value": "99.987%", "pages": [1]}]
        return json.dumps({
            "overview": {"title": title, "authors": authors, "date": None, "document_type": "document"},
            "sections": sections,
            "metrics": [{"label": d["label"], "value": d["value"], "pages": d["pages"]} for d in data[:6]],
            "not_stated": [],
        })

    raise ValueError(f"fake LLM does not know task {task!r}")


class FakeLLMClient:
    """Drop-in for LLMClient.complete_json used by unit tests (records every prompt it saw)."""

    def __init__(self, hallucinate: bool = False):
        self.config = SimpleNamespace(concurrency=2, model="fake-llm", max_output_tokens=4096)
        self.hallucinate = hallucinate
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, *, max_tokens: int | None = None) -> dict:
        self.calls += 1
        self.prompts.append((system, user))
        return json.loads(fake_respond(system, user, self.hallucinate))
