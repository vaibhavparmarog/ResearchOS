# ResearchOS

Upload any PDF — a research paper, report, manual or textbook chapter — and get an AI-generated,
page-referenced analysis of **that document only**.

**Live: https://researchos.duckdns.org**

> The application accepts a PDF supplied by the user at runtime and generates the analysis from that document.
> No PDF is bundled with the project, and nothing in the code or prompts depends on any particular document.

## What it does

1. Upload a PDF in the browser (drag & drop).
2. The server validates it and extracts the text page by page, in proper reading order (two-column papers included), using OCR when a page has no usable text layer.
3. The text goes through an LLM pipeline (single pass for short documents, map-reduce for long ones).
4. Every page reference, quote, metric, title and author in the model's output is **checked against the PDF** before it is shown.
5. You read the report (contents sidebar, verified quotes, key numbers, source list) and download it as PDF, Markdown, TXT or JSON.
6. **Cancel** stops a running analysis on the server immediately; **New PDF** discards everything about the previous one.

## Privacy: nothing is stored

- Nothing is written to disk: the container filesystem is read-only, uploads are spooled in RAM (`tmpfs`), nginx access logs are off.
- The PDF bytes are dropped as soon as the text is extracted, the extracted text as soon as the report exists.
- The finished report is sent to the browser once and purged from server memory ~30 s later; exports are rendered from the copy the browser sends back (`POST /api/export/{fmt}`), so the server never has to keep it.
- If the browser tab is closed, the analysis is cancelled (beacon on page hide, plus a heartbeat timeout).

## Architecture

```
researchos/
├── pdf/
│   ├── extractor.py      validation, per-page text, OCR fallback, headings, bibliography detection
│   ├── layout.py         reading order (stream order / column fallback), rotated-text removal, text-quality score, OCR
│   └── chunker.py        page-aware chunks with [[Page N]] markers
├── llm/
│   ├── client.py         multi-provider failover client (Groq, OpenRouter, NVIDIA, OpenAI, Anthropic, any OpenAI-compatible)
│   └── prompts.py        grounded prompts (instructions only, document text injected at runtime)
├── analysis/
│   ├── analyzer.py       map: structured notes per chunk (parallel)
│   ├── synthesizer.py    reduce: bounded hierarchical merging + final report
│   └── validator.py      grounding checks against the PDF
├── reports/              generator (orchestration), models, exporters (MD/TXT/JSON/PDF)
├── web/
│   ├── api.py            FastAPI: upload, status, cancel, stateless exports, static frontend, security headers
│   ├── jobs.py           in-memory jobs: cancel, heartbeat, one-time delivery + purge, queue
│   ├── render.py         report -> JSON for the browser
│   └── static/           index.html, app.js, styles.css (no framework, no CDN)
└── utils/                env config, error types, text helpers, safe Markdown renderer
deploy/                   nginx site config, EC2 setup script
Dockerfile, docker-compose.yml
tests/                    pytest suite (+ deterministic LLM test double)
```

## Pipeline

```
PDF -> validate -> per-page text (reading order, no rotated margin text)
    -> unreadable pages? -> OCR (Tesseract)
    -> clean (ligatures, hyphenation, running headers) -> drop bibliography from LLM input
    -> page-aware chunks
    -> short doc: one call on the full text | long doc: notes per chunk -> bounded merge -> final call
    -> validate every page/quote/metric/author against the PDF -> render / export
```

## LLM providers and failover

Configure one or more providers; they are tried in order. When a provider answers **429 (rate limit)**
the call moves to the next provider immediately and the limited one cools down for the time it asked for.
Invalid keys / no credit / unknown model disable that provider; "request too large" skips it for that call.
On Groq every model has its own rate-limit bucket, so listing several models multiplies throughput.

Requests are handled smartly:
- **Load balancing** - parallel chunk requests go to the provider/model with the fewest requests in flight, so chunk 1 goes to one model and chunk 2 to another at the same time.
- **Hedging** - if a call has not answered after `LLM_HEDGE_AFTER` seconds (default 40), the same request is also sent to an idle provider and the first answer wins.
- **Cut-off answers** are retried with a larger output budget (reasoning models spend tokens thinking).
- **Queueing** - limited workers, queue position shown to the user, max active jobs per client, uploads per hour per IP, nginx request rate limits.

| Variable | Example | Notes |
|---|---|---|
| `GROQ_API_KEY` | `gsk_...` | |
| `GROQ_MODEL` | `openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b` | comma list = several buckets |
| `OPENROUTER_API_KEY` | `sk-or-...` | |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b` | |
| `NVIDIA_API_KEY` | `nvapi-...` | |
| `NVIDIA_MODEL` | `nvidia/nemotron-3-super-120b-a12b,openai/gpt-oss-20b` | |
| `LLM_PROVIDERS` | `groq,openrouter,nvidia` | order (default: groq, openrouter, nvidia, openai, anthropic) |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` / `LLM_PROVIDER` | | one generic OpenAI-compatible (or `anthropic`) provider, tried first |
| `LLM_CONCURRENCY` / `LLM_MAX_OUTPUT_TOKENS` / `LLM_TIMEOUT` / `LLM_RETRY_WINDOW` / `LLM_HEDGE_AFTER` | `6` / `4000` / `120` / `240` / `40` | concurrency defaults to the number of provider/model slots |
| `SINGLE_PASS_CHARS` / `CHUNK_CHARS` / `NOTES_BUDGET_CHARS` | `45000` / `9000` / `9000` | whole papers go in one call to a provider that accepts the size |
| `MAX_PDF_MB` / `MAX_PDF_PAGES` | `25` / `300` | |
| `OCR_MODE` / `OCR_DPI` / `MAX_OCR_PAGES` / `OCR_LANGUAGE` | `auto` / `150` / `60` / `eng` | OCR needs Tesseract (included in the Docker image) |
| `SKIP_REFERENCES` | `true` | keep the bibliography out of the LLM input |
| `JOB_WORKERS` / `JOBS_PER_IP_PER_HOUR` / `MAX_PENDING_JOBS` / `MAX_ACTIVE_JOBS_PER_IP` | `2` / `30` / `20` / `2` | server limits |

Secrets go in `.env` (git-ignored, excluded from the Docker build context). See `.env.example`.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env        # add at least one API key
uvicorn researchos.web.api:app --port 8000
```

Open http://localhost:8000. Tests (no API key or network needed): `pip install -r requirements-dev.txt && python -m pytest tests -q`.

## Deploying on AWS EC2 (Docker + nginx + Let's Encrypt)

1. Ubuntu instance; security group allows inbound **22, 80, 443**; a DNS name pointing at its IP.
2. Copy your secrets: `scp .env ubuntu@<ip>:/opt/researchos/.env` (create the folder first).
3. On the server: `bash deploy/setup_ec2.sh v1.0.0 <your-domain> <email>` — installs Docker, nginx and certbot, checks out the release tag, builds and starts the container (bound to 127.0.0.1:8000), configures nginx and obtains an HTTPS certificate (auto-renewed by certbot's timer).

Updating to a new release: rerun the script with the new tag.

## Grounding / hallucination handling

Prompts restrict the model to the supplied text, treat the document as data (instructions inside a PDF are ignored), require verbatim quotes and "Not stated in the document" for missing facts, and separate document content from interpretation. The output is then **verified in code**:

- page numbers that do not exist (or have no text) are removed, including inline "page 999" mentions;
- every quote must occur verbatim on the cited page — otherwise its page is corrected, or the quote is removed;
- sections that claim document facts without any valid page are flagged;
- metric values must be found on the cited pages, otherwise re-attributed or flagged "not found";
- title, authors and date must appear in the opening pages or PDF metadata;
- every correction is listed under "Verification notes".

## Supported PDFs and limitations

Any PDF up to the size/page limits. Two-column layouts are read in column order; pages without a usable text layer (scans, fonts without a Unicode map such as the 1998 LeNet-5 PDF) are OCR'd, which is slow (~10–15 s per page) and can contain recognition errors. Figures are not analysed. Verification proves that quotes and numbers exist on the cited pages; it cannot prove that a paraphrase or an interpretation is correct. Free LLM tiers are slow for long documents.

## Security notes

Raw HTML in model output is never rendered; links are restricted to http(s); a strict Content-Security-Policy blocks inline and third-party scripts. Uploads are size-checked, jobs are addressed by random IDs and expire, and uploads per IP are rate-limited. The container runs as a non-root user and is only reachable through nginx.
