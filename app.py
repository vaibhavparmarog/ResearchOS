"""AI PDF Research Analyst - Streamlit UI.

    streamlit run app.py

The application accepts a PDF supplied by the user at runtime and generates the analysis from
that document. No PDF is bundled with, or read by, this app except the one the user uploads.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re

import streamlit as st

from pdf_analyst.llm.client import LLMClient
from pdf_analyst.reports import exporter
from pdf_analyst.reports.generator import extract_document, generate_report
from pdf_analyst.utils.config import LLMConfig, Limits, load_dotenv
from pdf_analyst.utils.errors import AnalystError, LLMConfigError
from pdf_analyst.utils.text import format_pages

load_dotenv()


def _load_streamlit_secrets() -> None:
    """On Streamlit Community Cloud, configuration comes from the app's Secrets; expose it as env vars."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(str(key), str(value))
    except Exception:       # no secrets configured (local run)
        pass


_load_streamlit_secrets()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pdf_analyst.app")

st.set_page_config(page_title="AI PDF Research Analyst", page_icon="\U0001F4C4", layout="centered")

EXPANDED_BY_DEFAULT = {"executive_summary", "key_findings", "key_takeaways"}
BASIS_BADGE = {
    "document": ":green-badge[From the document]",
    "interpretation": ":orange-badge[AI interpretation]",
    "mixed": ":blue-badge[Document + AI interpretation]",
}


# ----------------------------------------------------------------------------- state
def _state() -> dict:
    """Everything derived from the CURRENT upload lives here and is replaced wholesale on a new one."""
    return st.session_state.setdefault("analysis", {})


def _reset_analysis() -> None:
    st.session_state["analysis"] = {}


def _upload_another() -> None:
    st.session_state["upload_gen"] = st.session_state.get("upload_gen", 0) + 1
    _reset_analysis()


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", re.sub(r"\.pdf$", "", name, flags=re.I)).strip("_") or "document"


# ----------------------------------------------------------------------------- pipeline steps
def run_extraction(state: dict, data: bytes, filename: str) -> None:
    try:
        state["doc"] = extract_document(data, filename, Limits.from_env())
    except AnalystError as exc:
        log.info("extraction rejected %s: %s", filename, exc.detail or exc.user_message)
        state["error"] = exc.user_message
    except Exception:
        log.exception("unexpected extraction failure")
        state["error"] = "The PDF could not be processed. It may be damaged or use an unsupported format."


def run_analysis(state: dict) -> None:
    doc = state["doc"]
    bar = st.progress(0.0, text="Starting analysis")
    try:
        client = LLMClient(LLMConfig.from_env())
        try:
            state["report"] = generate_report(
                doc, client, Limits.from_env(), lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m)
            )
        finally:
            client.close()
        try:
            state["pdf_bytes"] = exporter.to_pdf(state["report"])
        except Exception:
            log.exception("PDF export failed")
            state["pdf_bytes"] = None
    except AnalystError as exc:
        log.warning("analysis failed: %s | %s", type(exc).__name__, exc.detail)
        state["error"] = exc.user_message
        state["error_is_config"] = isinstance(exc, LLMConfigError)
    except Exception:
        log.exception("unexpected analysis failure")
        state["error"] = "Something unexpected went wrong while generating the report. Please try again."
    finally:
        bar.empty()


# ----------------------------------------------------------------------------- rendering
def render_file_card(doc) -> None:
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.markdown(f"**{doc.filename}**")
    c2.metric("Size", f"{doc.size_bytes / 1024:.0f} KB" if doc.size_bytes < 1_048_576 else f"{doc.size_bytes / 1_048_576:.1f} MB")
    c3.metric("Pages", doc.page_count)
    st.caption(
        f"Extraction: ✅ text extracted from {len(doc.text_pages)} of {doc.page_count} pages "
        f"({doc.total_chars:,} characters)"
    )
    for w in doc.warnings:
        st.warning(w)


def render_report(report, doc, filename: str, pdf_bytes: bytes | None) -> None:
    o = report.overview
    st.header(o.title, anchor=False)
    st.markdown("\n".join(["| | |", "|---|---|"] + [f"| **{k}** | {v.replace('|', '/')} |" for k, v in exporter.overview_rows(report)]))

    for w in report.warnings:
        if w not in doc.warnings:
            st.warning(w)

    stem = _safe_stem(filename)
    cols = st.columns(4 if pdf_bytes else 3)
    cols[0].download_button("Markdown", exporter.to_markdown(report), f"{stem}_report.md", "text/markdown", on_click="ignore", use_container_width=True)
    cols[1].download_button("Text", exporter.to_text(report), f"{stem}_report.txt", "text/plain", on_click="ignore", use_container_width=True)
    cols[2].download_button("JSON", exporter.to_json(report), f"{stem}_report.json", "application/json", on_click="ignore", use_container_width=True)
    if pdf_bytes:
        cols[3].download_button("PDF", pdf_bytes, f"{stem}_report.pdf", "application/pdf", on_click="ignore", use_container_width=True)

    for s in report.sections:
        with st.expander(s.heading, expanded=s.key in EXPANDED_BY_DEFAULT):
            st.markdown(BASIS_BADGE[s.basis] + (f"  ·  **Source:** {format_pages(s.pages)}" if s.pages else ""))
            if not s.supported:
                st.warning("This section makes claims about the document but cites no verifiable page. Treat it with caution.")
            st.markdown(s.body_markdown)
            for e in s.evidence:
                st.markdown(f"> “{e.quote}”  \n> — *{format_pages([e.page])}, verified verbatim*")

    if report.metrics:
        with st.expander("Important data / metrics", expanded=True):
            st.markdown(":green-badge[Values checked against the PDF text]")
            per_row = 3
            for i in range(0, len(report.metrics), per_row):
                for col, m in zip(st.columns(per_row), report.metrics[i : i + per_row]):
                    with col:
                        st.metric(m.label, m.value)
                        st.caption(format_pages(m.pages) if m.verified else ":orange[⚠ not found in the PDF text — unverified]")

    if report.not_stated:
        with st.expander("Not stated in the document"):
            st.markdown("\n".join(f"- **{x}**: Not stated in the document." for x in report.not_stated))

    with st.expander(f"Source references ({len(report.references)} pages)"):
        for ref in report.references:
            st.markdown(f"**{format_pages([ref.page])}** — used in: {', '.join(ref.used_in)}")
            for q in ref.quotes:
                st.caption(f"“{q}”")

    with st.expander("Verification notes"):
        if report.validation_notes:
            st.markdown("Corrections applied to the AI output before display:")
            st.markdown("\n".join(f"- {n}" for n in report.validation_notes))
        else:
            st.markdown("All page references and quotes were checked against the PDF; nothing needed correcting.")
        m = report.meta
        st.caption(
            f"{m.get('model')} · {m.get('strategy')} · {m.get('chunks')} chunk(s) · "
            f"{m.get('llm_calls')} LLM call(s) · {m.get('seconds')}s · sha256 {str(m.get('sha256', ''))[:12]}…"
        )


# ----------------------------------------------------------------------------- page
def password_gate() -> bool:
    """Optional access control: if APP_PASSWORD is set, nothing runs (and no API calls are made) until it is entered.

    Protects the owner's LLM quota when the app is hosted publicly.
    """
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("authed"):
        return True
    entered = st.text_input("Password", type="password", help="Ask the app owner for the access password.")
    if entered:
        if hmac.compare_digest(entered.encode(), expected.encode()):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Incorrect password.")
    return False


def main() -> None:
    st.title("AI PDF Research Analyst")
    st.markdown("Upload a PDF and generate an AI-powered document analysis.")
    if not password_gate():
        return

    limits = Limits.from_env()
    if LLMConfig.missing_vars():
        st.info(
            "The AI service is not configured yet. Set **" + "**, **".join(LLMConfig.missing_vars())
            + "** (see README) and restart. You can still upload a PDF to check that it can be read."
        )

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key=f"uploader_{st.session_state.get('upload_gen', 0)}",
        help=f"Any text-based PDF, up to {limits.max_pdf_mb} MB and {limits.max_pages} pages.",
    )
    st.caption(f"Supported: PDF · up to {limits.max_pdf_mb} MB / {limits.max_pages} pages · text-based (scanned images are not supported)")

    if uploaded is None:
        _reset_analysis()
        st.markdown(
            "**How it works**\n\n1. Upload any PDF\n2. The text is extracted page by page\n"
            "3. The AI analyses *only* that document\n4. Inspect the report and export it, with page references for every claim"
        )
        return

    data = uploaded.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    state = _state()
    if state.get("sha") != sha:                 # a different file than before -> discard everything
        _reset_analysis()
        state = _state()
        state["sha"] = sha
        with st.spinner("Reading the PDF…"):
            run_extraction(state, data, uploaded.name)

    top = st.columns([3, 1])
    if top[1].button("Upload another PDF", use_container_width=True):
        _upload_another()
        st.rerun()

    if "doc" in state:
        render_file_card(state["doc"])

    if state.get("error") and "report" not in state:
        st.error(state["error"])
        if "doc" in state and not state.get("error_is_config"):
            if st.button("Try again"):
                state.pop("error", None)
                st.rerun()
        return

    if "report" not in state:
        run_analysis(state)
        if state.get("error"):
            st.rerun()      # show the error state above
    if "report" in state:
        render_report(state["report"], state["doc"], state["doc"].filename, state.get("pdf_bytes"))


main()
