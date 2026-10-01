"""FastAPI server: JSON API + static single-page frontend.

    uvicorn researchos.web.api:app --host 0.0.0.0 --port 8000

Privacy: nothing is written to disk. Uploads, extracted text and reports live in memory only
while a job runs; the finished report is handed to the browser once and then purged, and
exports are rendered from the copy the browser sends back (POST /api/export/{fmt}).
Run ONE worker process (jobs are kept in memory).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..llm.client import LLMClient, ProviderPool
from ..pdf.layout import ocr_available
from ..reports import exporter
from ..reports.models import Report
from ..utils.config import LLMConfig, Limits, load_dotenv, providers_from_env
from .jobs import BusyError, JobManager, RateLimitedError, TooManyActiveError
from .render import report_payload

load_dotenv()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("researchos.web")

STATIC = Path(__file__).parent / "static"
MAX_EXPORT_BODY = 3 * 1024 * 1024
EXPORTS = {
    "md": ("text/markdown; charset=utf-8", exporter.to_markdown),
    "txt": ("text/plain; charset=utf-8", exporter.to_text),
    "json": ("application/json", exporter.to_json),
    "pdf": ("application/pdf", exporter.to_pdf),
}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _default_client_factory() -> Callable[[threading.Event], LLMClient]:
    """One shared provider pool per process; each job gets a light client (with its own cancel signal)."""
    pool: dict[str, ProviderPool] = {}
    lock = threading.Lock()

    def factory(cancel: threading.Event) -> LLMClient:
        with lock:
            if "p" not in pool:
                pool["p"] = ProviderPool(LLMConfig.from_env())   # raises LLMConfigError -> shown to the user
        return LLMClient(pool=pool["p"], cancel_event=cancel)

    return factory


def create_app(
    client_factory: Callable[[threading.Event], object] | None = None,
    limits_factory: Callable[[], Limits] = Limits.from_env,
) -> FastAPI:
    jobs = JobManager(
        client_factory or _default_client_factory(),
        limits_factory,
        workers=_env_int("JOB_WORKERS", 2),
        max_pending=_env_int("MAX_PENDING_JOBS", 20),
        per_ip_per_hour=_env_int("JOBS_PER_IP_PER_HOUR", 30),
        max_active_per_ip=_env_int("MAX_ACTIVE_JOBS_PER_IP", 2),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="ResearchOS", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.jobs = jobs

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # ------------------------------------------------------------------ api
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/config")
    def config() -> dict:
        limits = limits_factory()
        providers = providers_from_env() if client_factory is None else []
        return {
            "version": __version__,
            "max_pdf_mb": limits.max_pdf_mb,
            "max_pages": limits.max_pages,
            "llm_configured": bool(providers) or client_factory is not None,
            "providers": sorted({p.name.split("#")[0] for p in providers}),
            "ocr": ocr_available(),
            "busy": jobs.active_count(),
        }

    @app.post("/api/jobs", status_code=202)
    async def create_job(request: Request, file: UploadFile | None = File(None)) -> dict:
        limits = limits_factory()
        if file is None or not file.filename:
            raise HTTPException(400, "No file was uploaded. Please choose a PDF.")
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "That file is not a PDF. Please upload a file with a .pdf extension.")
        max_bytes = limits.max_pdf_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        await file.close()
        if len(data) > max_bytes:
            raise HTTPException(413, f"The file is larger than the {limits.max_pdf_mb} MB limit.")
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        if b"%PDF-" not in data[:1024]:
            raise HTTPException(400, "The file has a .pdf extension but its contents are not a PDF document.")
        ip = request.client.host if request.client else "-"
        try:
            job = jobs.submit(data, Path(file.filename).name[:200], ip)
        except RateLimitedError:
            raise HTTPException(429, "Too many uploads from your address in the last hour. Please try again later.")
        except TooManyActiveError:
            raise HTTPException(429, "You already have analyses running. Wait for them to finish or cancel one.")
        except BusyError:
            raise HTTPException(503, "The server is busy with other documents. Please try again in a few minutes.")
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "This analysis no longer exists. Please upload the PDF again.")
        out = job.public()
        if job.status == "done" and job.report is not None:
            out["report"] = report_payload(job.report)
            out["report_data"] = job.report.to_dict()       # the browser keeps this for exports
            jobs.mark_delivered(job)                       # purged from the server shortly after
        return out

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        return {"cancelled": jobs.cancel(job_id)}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        return {"deleted": jobs.cancel(job_id)}

    @app.post("/api/export/{fmt}")
    async def export(fmt: str, request: Request) -> Response:
        """Render an export from the report the browser holds; nothing is looked up or stored."""
        if fmt not in EXPORTS:
            raise HTTPException(400, "Unknown export format.")
        body = await request.body()
        if len(body) > MAX_EXPORT_BODY:
            raise HTTPException(413, "Report too large to export.")
        try:
            report = Report.from_dict(json.loads(body))
        except (ValueError, TypeError, KeyError, AttributeError):
            raise HTTPException(400, "The report data is not valid.")
        media, render = EXPORTS[fmt]
        out = render(report)
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(report.overview.filename).stem)[:80] or "document"
        return Response(
            out if isinstance(out, bytes) else out.encode("utf-8"),
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{stem}_report.{fmt}"'},
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse({"error": "Something went wrong on the server. Please try again."}, status_code=500)

    return app


app = create_app()
