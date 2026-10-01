"use strict";
// ResearchOS frontend: upload -> poll job -> render report. No framework, no external requests.

const $ = (id) => document.getElementById(id);
const OPEN_BY_DEFAULT = new Set(["executive_summary", "key_findings", "key_takeaways", "contributions"]);
let currentJob = null;
let currentFile = null;
let pollTimer = null;
let maxMb = 25;

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

function humanSize(n) {
  return n < 1048576 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1048576).toFixed(1)} MB`;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = {};
  try { body = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) throw new Error(body.error || `Request failed (HTTP ${res.status})`);
  return body;
}

// ------------------------------------------------------------------ setup
async function loadConfig() {
  try {
    const cfg = await api("/api/config");
    maxMb = cfg.max_pdf_mb;
    $("version").textContent = `v${cfg.version}`;
    $("limits-hint").textContent = `PDF up to ${cfg.max_pdf_mb} MB / ${cfg.max_pages} pages` + (cfg.ocr ? " · scanned PDFs via OCR" : "");
    if (!cfg.llm_configured) {
      const b = $("config-banner");
      b.textContent = "The AI service is not configured on this server yet (set GROQ_API_KEY, OPENROUTER_API_KEY or NVIDIA_API_KEY). PDFs can still be uploaded to check they are readable.";
      b.hidden = false;
    }
  } catch (_) { /* server unreachable: the upload will show the error */ }
}

function setupDrop() {
  const drop = $("drop"), input = $("file");
  input.addEventListener("change", () => input.files[0] && start(input.files[0]));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) start(f); });
  $("another").addEventListener("click", reset);
  $("retry").addEventListener("click", () => currentFile && start(currentFile));
}

// ------------------------------------------------------------------ flow
function showUploadError(msg) {
  const e = $("upload-error");
  e.textContent = msg;
  e.hidden = false;
}

async function start(file) {
  $("upload-error").hidden = true;
  if (!file.name.toLowerCase().endsWith(".pdf")) return showUploadError("That file is not a PDF. Please choose a .pdf file.");
  if (file.size > maxMb * 1048576) return showUploadError(`The file is ${humanSize(file.size)}; the limit is ${maxMb} MB.`);
  await discardCurrent();
  currentFile = file;
  $("upload-view").hidden = true;
  $("hero").hidden = true;
  $("job-view").hidden = false;
  $("report").hidden = true;
  $("report").replaceChildren();
  $("job-error").hidden = true;
  $("progress-card").hidden = false;
  $("doc-warnings").replaceChildren();
  $("f-name").textContent = file.name;
  $("f-facts").textContent = `${humanSize(file.size)} · uploading`;
  setProgress(0.01, "Uploading");
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const job = await api("/api/jobs", { method: "POST", body: form });
    currentJob = job.id;
    render(job);
    poll();
  } catch (err) {
    fail(err.message, true);
  }
}

function poll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (!currentJob) return;
    const id = currentJob;
    try {
      const job = await api(`/api/jobs/${id}`);
      if (id !== currentJob) return;            // user moved on to another file
      render(job);
      if (job.status !== "done" && job.status !== "error") poll();
    } catch (err) {
      if (id === currentJob) fail(err.message, true);
    }
  }, 1200);
}

async function discardCurrent() {
  clearTimeout(pollTimer);
  const id = currentJob;
  currentJob = null;
  if (id) { try { await fetch(`/api/jobs/${id}`, { method: "DELETE" }); } catch (_) { /* ignore */ } }
}

async function reset() {
  await discardCurrent();
  currentFile = null;
  $("file").value = "";
  $("job-view").hidden = true;
  $("report").replaceChildren();
  $("upload-view").hidden = false;
  $("hero").hidden = false;
  $("drop").focus();
}

function setProgress(fraction, message) {
  const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  $("p-bar").style.width = `${pct}%`;
  $("p-pct").textContent = `${pct}%`;
  $("p-msg").textContent = message || "";
}

function fail(message, retryable) {
  $("progress-card").hidden = true;
  $("job-error-text").textContent = message;
  $("retry").hidden = !retryable;
  $("job-error").hidden = false;
}

function render(job) {
  const d = job.document;
  if (d) {
    const facts = [`${humanSize(d.size_bytes)}`, `${d.pages} page${d.pages === 1 ? "" : "s"}`, `text on ${d.text_pages}/${d.pages}`];
    if (d.ocr_pages.length) facts.push(`${d.ocr_pages.length} page(s) via OCR`);
    if (d.reference_pages.length) facts.push(`bibliography p.${d.reference_pages[0]}–${d.reference_pages[d.reference_pages.length - 1]} skipped`);
    $("f-facts").textContent = facts.join(" · ");
    const w = $("doc-warnings");
    if (!w.childElementCount) d.warnings.forEach((msg) => w.append(el("div", { class: "banner warn", text: msg })));
  }
  if (job.status === "error") return fail(job.error, !job.error_is_config);
  setProgress(job.progress, job.message);
  if (job.status === "done" && job.report) {
    $("progress-card").hidden = true;
    renderReport(job.id, job.report);
  }
}

// ------------------------------------------------------------------ report
function section(title, key, open, ...body) {
  const d = el("details", { class: "section" }, el("summary", { text: title }), el("div", { class: "section-body" }, ...body));
  if (open) d.open = true;
  if (key) d.dataset.key = key;
  return d;
}

function renderReport(jobId, r) {
  const root = $("report");
  root.replaceChildren();

  root.append(el("h2", { class: "title", text: r.title }));
  const table = el("table", { class: "overview" });
  r.overview.forEach((row) => table.append(el("tr", {}, el("th", { text: row.label }), el("td", { text: row.value }))));
  root.append(el("div", { class: "card" }, table));

  const exports = el("div", { class: "exports" }, el("span", { text: "Download:" }));
  [["md", "Markdown"], ["txt", "Text"], ["pdf", "PDF"], ["json", "JSON"]].forEach(([fmt, label]) =>
    exports.append(el("a", { class: "btn secondary", href: `/api/jobs/${jobId}/export/${fmt}`, download: "", text: label })));
  root.append(exports);

  r.warnings.filter((w) => !document.querySelector("#doc-warnings").textContent.includes(w))
    .forEach((w) => root.append(el("div", { class: "banner warn", text: w })));

  for (const s of r.sections) {
    const meta = el("div", { class: "meta-line" }, el("span", { class: `badge ${s.basis}`, text: s.basis_label }));
    if (s.source) meta.append(el("span", { text: `Source: ${s.source}` }));
    if (!s.supported) meta.append(el("span", { class: "badge bad", text: "No verifiable page reference" }));
    const content = el("div", { class: "content" });
    content.innerHTML = s.html;                  // server-rendered with raw HTML disabled (see web/render.py)
    const quotes = s.evidence.map((e) => el("blockquote", { class: "quote" }, `“${e.quote}”`, el("cite", { text: `${e.source} · verified verbatim in the PDF` })));
    root.append(section(s.heading, s.key, OPEN_BY_DEFAULT.has(s.key), meta, content, ...quotes));
  }

  if (r.metrics.length) {
    const grid = el("div", { class: "metrics" });
    r.metrics.forEach((m) => grid.append(el("div", { class: `metric${m.verified ? "" : " unverified"}` },
      el("div", { class: "label", text: m.label }), el("div", { class: "value", text: m.value }),
      el("div", { class: "src", text: m.verified ? m.source : "⚠ not found in the PDF text" }))));
    root.append(section("Important data / metrics", "metrics", true, el("p", { class: "small", text: "Each value was checked against the text of the cited page." }), grid));
  }

  if (r.not_stated.length) {
    root.append(section("Not stated in the document", "not_stated", false,
      el("ul", {}, r.not_stated.map((x) => el("li", { text: `${x}: not stated in the document.` })))));
  }

  const refs = r.references.map((ref) => el("div", { class: "ref" }, el("strong", { text: ref.source }), ` — used in: ${ref.used_in.join(", ")}`,
    ...ref.quotes.map((q) => el("q", { text: q }))));
  root.append(section(`Source references (${r.references.length} pages)`, "refs", false, ...(refs.length ? refs : [el("p", { text: "No page references could be verified." })])));

  const notes = r.validation_notes.length
    ? [el("p", { text: "Corrections applied to the AI output before display:" }), el("ul", {}, r.validation_notes.map((n) => el("li", { text: n })))]
    : [el("p", { text: "All page references and quotes were checked against the PDF; nothing needed correcting." })];
  const m = r.meta;
  notes.push(el("p", { class: "small", text: `${m.model} · ${m.strategy} · ${m.chunks} chunk(s) · ${m.llm_calls} LLM call(s) · ${m.seconds}s` }));
  root.append(section("Verification notes", "verification", false, ...notes));

  root.hidden = false;
}

document.addEventListener("DOMContentLoaded", () => { setupDrop(); loadConfig(); });
