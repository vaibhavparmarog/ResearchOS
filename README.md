# AI PDF Research Analyst

Upload a PDF and get an AI-generated, page-referenced analysis of it.

> **The application accepts a PDF supplied by the user at runtime and generates the analysis from that document.**
> No PDF is bundled with the project, no PDF is processed automatically, and nothing in the code or prompts
> depends on the content of any particular document.

## What it does

1. You open the app and upload any text-based PDF.
2. The app validates it, extracts the text page by page and shows file name, size, page count and extraction status.
3. The extracted text (and only that) goes through an LLM analysis pipeline.
4. You get a structured report with collapsible sections, page references, verbatim supporting quotes, metric cards and a verification log.
5. You export it as Markdown, plain text, PDF or JSON.
6. **Upload another PDF** clears everything and produces a completely new report; nothing is reused from the previous document.

## Architecture

```
app.py                       Streamlit UI (upload, progress, report rendering, downloads)
pdf_analyst/
├── pdf/
│   ├── extractor.py         validation, PyMuPDF text extraction, headings/outline, warnings
│   └── chunker.py           page-aware chunks with [[Page N]] markers
├── llm/
│   ├── client.py            httpx client: OpenAI-compatible or Anthropic API, retries, JSON repair
│   └── prompts.py           system prompts + prompt builders (instructions only, no document facts)
├── analysis/
│   ├── analyzer.py          map stage: notes per chunk (parallel)
│   ├── synthesizer.py       reduce (hierarchical condensing) + final report call
│   └── validator.py         grounding checks on everything the LLM returns
├── reports/
│   ├── generator.py         orchestration: PDF bytes -> validated Report
│   ├── models.py            Report / Section / Metric dataclasses
│   └── exporter.py          Markdown, TXT, JSON, PDF export
└── utils/                   config (env vars), error types, text cleaning/matching
tests/                       pytest suite (+ a deterministic LLM test double)
```

## How it works

```
PDF bytes -> validate -> extract text per page -> clean -> page-aware chunks
          -> [short doc]  final report from the full text
          -> [long doc]   notes per chunk -> hierarchical merge -> final report from merged notes
          -> validate against the PDF -> render / export
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Environment Variables

Set them in your shell or in a git-ignored `.env` file (see `.env.example`).

| Variable | Required | Description |
|---|---|---|
| `LLM_API_KEY` | yes | API key for the LLM provider |
| `LLM_MODEL` | yes | Model name, e.g. `gpt-4o-mini` or a Claude model id |
| `LLM_BASE_URL` | no | Default `https://api.openai.com/v1`. Any OpenAI-compatible endpoint works (OpenAI, Groq, OpenRouter, Gemini's OpenAI endpoint, Ollama `http://localhost:11434/v1`, vLLM ...) |
| `LLM_PROVIDER` | no | `openai` (default) or `anthropic` (Messages API; base URL defaults to `https://api.anthropic.com`) |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` / `LLM_CONCURRENCY` / `LLM_MAX_OUTPUT_TOKENS` / `LLM_JSON_MODE` | no | Request tuning (defaults 120 s / 3 / 3 / 4096 / true) |
| `MAX_PDF_MB` / `MAX_PDF_PAGES` | no | Limits (defaults 25 MB / 300 pages). Streamlit's own cap is in `.streamlit/config.toml`. |
| `CHUNK_CHARS` / `SINGLE_PASS_CHARS` / `NOTES_BUDGET_CHARS` | no | Chunking thresholds (defaults 12000 / 40000 / 30000 characters) |

**Example: Groq** (verified with `openai/gpt-oss-120b`; check `https://api.groq.com/openai/v1/models` for the models your key can use):

```
LLM_API_KEY=<your key>
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Free tiers cap tokens per minute (8,000 for that model on Groq's free tier). For such plans also set
`LLM_CONCURRENCY=1`, `LLM_MAX_OUTPUT_TOKENS=3500`, `SINGLE_PASS_CHARS=10000`, `CHUNK_CHARS=6000`, `NOTES_BUDGET_CHARS=6000`.
The client honours the provider's "try again in Ns" hint, so long documents still finish, just slower (a 45-page manual took about 5 minutes there).

If the key or model is missing the app does not crash: it still reads the PDF and then shows which variables to set.

## Running Locally

```bash
streamlit run app.py
```

Run the tests (no API key or network needed):

```bash
python -m pytest tests -q
```

## Supported PDFs

Any PDF with a selectable text layer: papers, reports, manuals, textbook chapters, business documents, and so on. The
report structure adapts to the document, so methodology/results sections only appear when the document has them.
Password-protected, corrupted, empty and image-only (scanned) PDFs are rejected with a clear message. **OCR is not implemented.**

## Report Generation Pipeline

1. **Ingestion** (`pdf/extractor.py`): magic-byte and extension check, size/page limits, encryption check, PyMuPDF extraction per page, headings from the PDF outline or from font sizes, per-page empty detection with warnings.
2. **Cleaning** (`utils/text.py`): ligatures, smart quotes, line-break hyphenation, and running headers/footers (repeated edge lines only).
3. **Chunking** (`pdf/chunker.py`): whole pages are packed into chunks of ~12k characters; oversized pages are split. Every page carries an explicit `[[Page N]]` marker, so page provenance survives.
4. **Analysis** (`analysis/`): documents up to ~40k characters are sent to the final call in full. Longer documents use map-reduce: structured notes per chunk (parallel), hierarchical condensing until they fit a budget, then the final synthesis, which also receives the raw opening pages for title/author/date.
5. **Validation** (`analysis/validator.py`): see below.
6. **Rendering/export** (`reports/`).

## Grounding / Hallucination Handling

Prompts tell the model to use only the supplied text, to treat the document as data (instructions inside a PDF are not followed), to say "Not stated in the document" for missing information, to copy quotes verbatim, and to label its own inference. Because prompts cannot guarantee that, the output is **verified in code against the PDF** before display:

- Page numbers that do not exist (or have no text) are dropped, including inline mentions like "page 999" in the prose.
- Each supporting quote must occur verbatim on the cited page; if it occurs on another page the page is corrected, otherwise the quote is removed.
- Sections that claim to restate the document but have no valid page are flagged as unsupported.
- Metric values must be found in the cited pages (otherwise re-attributed or flagged "not found in the PDF text").
- Title, authors and date must appear in the opening pages or PDF metadata, otherwise they are removed and shown as "Not stated in the document."
- Sections are labelled *From the document*, *AI interpretation*, or both; inferred limitations are always labelled as interpretation.
- The Source References list is built from verified quotes and pages only, and every correction is listed under "Verification notes".

## Limitations

- **Verification is mechanical**: it proves that quotes and numbers exist on the cited pages. It cannot prove that a paraphrase or an interpretation is correct. Read AI-interpretation sections critically.
- No OCR; tables and multi-column layouts depend on PyMuPDF's reading order; figures and images are not analysed.
- Section-level page references (not sentence-level) unless a verified quote is attached.
- Long documents are condensed, so fine detail can be lost; quality depends on the chosen model and its context window.
- The pipeline was built and tested with a deterministic test double; see the note on verification below. Real-model output quality still needs a check with your own API key.

## Example Workflow

1. `set LLM_API_KEY=...`, `set LLM_MODEL=...`, then `streamlit run app.py`.
2. Upload paper A (say a 10-page research article): read its report, expand *Key Findings* and check the quoted passages on the cited pages.
3. Click **Upload another PDF**, upload an unrelated document (a report, a manual): the previous analysis disappears and a new report about the new file is generated.
4. Download the Markdown/PDF export of the current report.

## Testing

`tests/` generates its own throw-away PDFs at run time (`tests/pdf_factory.py`); none are committed. The suite covers ingestion errors (corrupt, empty, image-only, encrypted, wrong type, too large), page-aware chunking, A-then-B dynamic behaviour with a leak check (including reuse of one client), long-document map-reduce, the validator against a deliberately hallucinating model, LLM error handling (missing key, 429 retry, 401, 5xx, timeout, malformed JSON) over a real HTTP round trip, and all exports.

`tests/fake_llm.py` is **not a model**: it derives its answers mechanically from the prompt text, so tests can prove the uploaded content really flows into the LLM and that the validator works. `python -m tests.fake_llm_server 8765` exposes it as an OpenAI-compatible server for offline UI checks (`LLM_BASE_URL=http://127.0.0.1:8765/v1`).
