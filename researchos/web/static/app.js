"use strict";
// ResearchOS frontend: upload -> poll job -> render report -> export from the browser's copy.
// No framework, no third-party requests. DOM is built with textContent; the only HTML inserted
// is section bodies that the server renders with raw HTML disabled (researchos/utils/safe_markdown.py).

const $ = (id) => document.getElementById(id);
const OPEN_BY_DEFAULT = new Set(["executive_summary", "key_findings", "key_takeaways"]);
const state = { job: null, file: null, timer: null, started: 0, reportData: null, maxMb: 25,
  ticker: null, shown: 0, server: 0, status: "queued", msgSince: 0, msg: "" };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "ic");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

const humanSize = (n) => (n < 1048576 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1048576).toFixed(1)} MB`);
const elapsed = () => {
  const s = Math.round((Date.now() - state.started) / 1000);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
};

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = {};
  try { body = await res.json(); } catch (_) { /* not JSON */ }
  if (!res.ok) throw new Error(body.error || `Request failed (HTTP ${res.status})`);
  return body;
}

// ------------------------------------------------------------------ setup
async function loadConfig() {
  try {
    const cfg = await api("/api/config");
    state.maxMb = cfg.max_pdf_mb;
    $("version").textContent = `v${cfg.version}`;
    $("limits-hint").textContent = `PDF up to ${cfg.max_pdf_mb} MB · ${cfg.max_pages} pages` + (cfg.ocr ? " · scans supported (OCR)" : "");
    if (!cfg.llm_configured) {
      const b = $("config-banner");
      b.textContent = "The AI service is not configured on this server yet. PDFs can still be uploaded to check that they are readable.";
      b.hidden = false;
    }
  } catch (_) { /* the upload will surface any server problem */ }
}

function setupEvents() {
  const drop = $("drop"), input = $("file");
  input.addEventListener("change", () => input.files[0] && start(input.files[0]));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) start(f); });
  $("another").addEventListener("click", reset);
  $("cancel").addEventListener("click", cancel);
  $("retry").addEventListener("click", () => state.file && start(state.file));
  // Closing the tab stops the analysis on the server right away (no wasted work, nothing kept).
  window.addEventListener("pagehide", () => { if (state.job) navigator.sendBeacon(`/api/jobs/${state.job}/cancel`); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && state.job) poll(0); });
}

// ------------------------------------------------------------------ flow
function showUploadError(msg) {
  const e = $("upload-error");
  e.textContent = msg;
  e.hidden = false;
}

function showView(view) {
  $("upload-view").hidden = view !== "upload";
  $("hero").hidden = view !== "upload";
  $("job-view").hidden = view !== "job";
}

async function start(file) {
  $("upload-error").hidden = true;
  if (!file.name.toLowerCase().endsWith(".pdf")) return showUploadError("That file is not a PDF. Please choose a .pdf file.");
  if (file.size > state.maxMb * 1048576) return showUploadError(`The file is ${humanSize(file.size)}; the limit is ${state.maxMb} MB.`);
  await stopCurrent();
  state.file = file;
  state.started = Date.now();
  state.reportData = null;
  state.shown = 0;
  state.server = 0;
  startTicker();
  showView("job");
  $("report-layout").hidden = true;
  $("report").replaceChildren();
  $("toc").replaceChildren();
  $("job-error").hidden = true;
  $("progress-card").hidden = false;
  $("doc-warnings").replaceChildren();
  $("cancel").hidden = false;
  $("another").hidden = true;
  $("f-name").textContent = file.name;
  $("f-facts").textContent = `${humanSize(file.size)} · uploading`;
  setProgress("queued", 0.01, "Uploading");
  window.scrollTo({ top: 0 });
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const job = await api("/api/jobs", { method: "POST", body: form });
    state.job = job.id;
    render(job);
    poll();
  } catch (err) {
    fail(err.message, true);
  }
}

function poll(delay = 1200) {
  clearTimeout(state.timer);
  state.timer = setTimeout(async () => {
    const id = state.job;
    if (!id) return;
    try {
      const job = await api(`/api/jobs/${id}`);
      if (id !== state.job) return;            // the user moved on
      render(job);
      if (["done", "error", "cancelled"].includes(job.status)) state.job = null;
      else poll();
    } catch (err) {
      if (id === state.job) { state.job = null; fail(err.message, true); }
    }
  }, delay);
}

async function stopCurrent() {
  clearTimeout(state.timer);
  stopTicker();
  const id = state.job;
  state.job = null;
  if (id) { try { await fetch(`/api/jobs/${id}/cancel`, { method: "POST" }); } catch (_) { /* ignore */ } }
}

async function cancel() {
  await stopCurrent();
  reset();
}

async function reset() {
  await stopCurrent();
  state.file = null;
  state.reportData = null;
  $("file").value = "";
  $("report").replaceChildren();
  $("toc").replaceChildren();
  showView("upload");
  $("drop").focus();
}

const STEP_OF = { queued: 0, reading: 1, analyzing: 2, done: 4, error: -1, cancelled: -1 };

function setProgress(status, fraction, message) {
  let step = STEP_OF[status] ?? 0;
  if (status === "analyzing" && fraction >= 0.9) step = 3;
  document.querySelectorAll("#steps li").forEach((li) => {
    const i = Number(li.dataset.step);
    li.classList.toggle("done", i < step);
    li.classList.toggle("active", i === step);
  });
  state.status = status;
  state.server = fraction;
  state.shown = Math.max(state.shown, fraction);
  if (message !== state.msg) { state.msg = message || ""; state.msgSince = Date.now(); }
  draw();
}

// Runs every second while a job is active: ticking timer, a bar that keeps moving gently during
// long AI calls (never past the next real milestone), and a reassurance note for slow steps.
function tick() {
  if (state.status === "analyzing" || state.status === "reading") {
    const ceiling = Math.min(0.97, state.server + (state.server < 0.8 ? 0.12 : 0.08));
    state.shown = Math.min(ceiling, state.shown + (ceiling - state.shown) * 0.03);
  }
  draw();
}

function draw() {
  $("p-bar").style.width = `${Math.round(Math.max(0.02, Math.min(1, state.shown)) * 100)}%`;
  const slow = Date.now() - state.msgSince > 20000 && ["analyzing", "reading"].includes(state.status);
  $("p-msg").textContent = state.msg + (slow ? " · still working, long papers can take a couple of minutes (keep this tab open)" : "");
  $("p-time").textContent = state.started ? elapsed() : "";
}

function startTicker() { clearInterval(state.ticker); state.ticker = setInterval(tick, 1000); }
function stopTicker() { clearInterval(state.ticker); state.ticker = null; }

function fail(message, retryable) {
  stopTicker();
  $("progress-card").hidden = true;
  $("cancel").hidden = true;
  $("another").hidden = false;
  $("job-error-text").textContent = message;
  $("retry").hidden = !retryable;
  $("job-error").hidden = false;
}

function render(job) {
  const d = job.document;
  if (d) {
    const facts = [humanSize(d.size_bytes), `${d.pages} page${d.pages === 1 ? "" : "s"}`];
    if (d.text_pages < d.pages) facts.push(`text on ${d.text_pages}/${d.pages}`);
    if (d.ocr_pages.length) facts.push(`${d.ocr_pages.length} page(s) via OCR`);
    if (d.reference_pages.length) facts.push("bibliography skipped");
    $("f-facts").textContent = facts.join(" · ");
    const w = $("doc-warnings");
    if (!w.childElementCount) d.warnings.forEach((msg) => w.append(el("div", { class: "banner warn", text: msg })));
  }
  if (job.status === "cancelled") return reset();
  if (job.status === "error") return fail(job.error, !job.error_is_config);
  setProgress(job.status, job.progress, job.message);
  if (job.status === "done" && job.report) {
    stopTicker();
    state.reportData = job.report_data;          // kept only in this tab; the server has deleted it
    $("progress-card").hidden = true;
    $("cancel").hidden = true;
    $("another").hidden = false;
    renderReport(job.report);
  }
}

// ------------------------------------------------------------------ exports (stateless)
async function download(fmt, button) {
  if (!state.reportData) return;
  const label = button.lastChild.textContent;
  button.disabled = true;
  button.lastChild.textContent = "…";
  try {
    const res = await fetch(`/api/export/${fmt}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.reportData),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || "Export failed");
    const blob = await res.blob();
    const name = (res.headers.get("content-disposition") || "").match(/filename="([^"]+)"/)?.[1] || `report.${fmt}`;
    const a = el("a", { href: URL.createObjectURL(blob), download: name });
    document.body.append(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
    button.lastChild.textContent = label;
  }
}

// ------------------------------------------------------------------ report
function sectionCard(id, title, basis, open, ...body) {
  const d = el("details", { class: `section ${basis || ""}`, id }, el("summary", { text: title }), el("div", { class: "section-body" }, ...body));
  if (open) d.open = true;
  return d;
}

function tocLink(id, title, basis) {
  const a = el("a", { href: `#${id}` }, el("span", { class: `tdot ${basis || ""}` }), title);
  a.addEventListener("click", () => { const d = $(id); if (d) d.open = true; });
  return a;
}

function renderReport(r) {
  const root = $("report"), toc = $("toc");
  root.replaceChildren();
  toc.replaceChildren(el("div", { class: "toc-title", text: "Contents" }));

  // title card with facts + downloads
  const facts = el("dl", { class: "facts" });
  r.overview.filter((row) => row.label !== "Title").forEach((row) => facts.append(el("div", {}, el("dt", { text: row.label }), el("dd", { text: row.value }))));
  const toolbar = el("div", { class: "toolbar" }, el("span", { class: "label", text: "Download" }));
  [["pdf", "PDF"], ["md", "Markdown"], ["txt", "Text"], ["json", "JSON"]].forEach(([fmt, label]) => {
    const b = el("button", { class: "btn ghost", type: "button" }, icon("download"), label);
    b.addEventListener("click", () => download(fmt, b));
    toolbar.append(b);
  });
  root.append(el("div", { class: "card title-card", id: "overview" }, el("h2", { text: r.title }), facts, toolbar));
  toc.append(tocLink("overview", "Overview"));

  const shown = new Set(Array.from($("doc-warnings").children, (n) => n.textContent));
  r.warnings.filter((w) => !shown.has(w)).forEach((w) => root.append(el("div", { class: "banner warn", text: w })));

  r.sections.forEach((s, i) => {
    const id = `s-${i}`;
    const meta = el("div", { class: "meta-line" }, el("span", { class: `badge ${s.basis}`, text: s.basis_label }));
    if (s.source) meta.append(el("span", { class: "source" }, icon("pin"), s.source));
    if (!s.supported) meta.append(el("span", { class: "badge bad", text: "No verifiable page reference" }));
    const content = el("div", { class: "content" });
    content.innerHTML = s.html;                 // sanitised server-side (raw HTML disabled, links restricted)
    const quotes = s.evidence.length
      ? el("div", { class: "quotes" }, s.evidence.map((e) => el("blockquote", { class: "quote" }, icon("quote"), e.quote,
          el("cite", { text: `${e.source} · found verbatim in the PDF` }))))
      : null;
    root.append(sectionCard(id, s.heading, s.basis, OPEN_BY_DEFAULT.has(s.key) || i === 0, meta, content, quotes));
    toc.append(tocLink(id, s.heading, s.basis));
  });

  if (r.metrics.length) {
    const grid = el("div", { class: "metrics" }, r.metrics.map((m) => el("div", { class: `metric${m.verified ? "" : " unverified"}` },
      el("div", { class: "value", text: m.value }), el("div", { class: "label", text: m.label }),
      el("div", { class: "src", text: m.verified ? m.source : "⚠ not found in the PDF text" }))));
    root.append(sectionCard("metrics", "Key numbers", "document", true,
      el("p", { class: "small", text: "Each value was checked against the text of the cited page." }), grid));
    toc.append(tocLink("metrics", "Key numbers", "document"));
  }

  if (r.not_stated.length) {
    root.append(sectionCard("not-stated", "Not stated in the document", "", false,
      el("ul", {}, r.not_stated.map((x) => el("li", { text: x })))));
    toc.append(tocLink("not-stated", "Not stated"));
  }

  const refs = r.references.map((ref) => el("div", { class: "ref" }, el("strong", { text: ref.source }), ` — ${ref.used_in.join(", ")}`,
    ref.quotes.map((q) => el("q", { text: q }))));
  root.append(sectionCard("sources", `Sources (${r.references.length} pages)`, "", false,
    ...(refs.length ? refs : [el("p", { text: "No page references could be verified." })])));
  toc.append(tocLink("sources", "Sources"));

  const m = r.meta;
  const notes = r.validation_notes.length
    ? [el("p", { text: "Corrections applied to the AI output before display:" }), el("ul", {}, r.validation_notes.map((n) => el("li", { text: n })))]
    : [el("p", { text: "Every page reference and quote was checked against the PDF; nothing needed correcting." })];
  notes.push(el("p", { class: "small", text: `${m.model} · ${m.strategy} · ${m.llm_calls} AI call(s) · ${elapsed()} total` }));
  root.append(sectionCard("verification", "Verification", "", false, ...notes));
  toc.append(tocLink("verification", "Verification"));

  $("report-layout").hidden = false;
}

document.addEventListener("DOMContentLoaded", () => { setupEvents(); loadConfig(); });
