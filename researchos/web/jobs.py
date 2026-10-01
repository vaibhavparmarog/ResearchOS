"""In-memory analysis jobs: one upload = one job, processed on a small worker pool.

Jobs live only in this process's memory (run the server with a single worker process) and are
dropped after JOB_TTL_SECONDS or when the user starts over. Job ids are random UUIDs, so one
visitor cannot read another visitor's report.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from ..pdf.chunker import analysis_chars
from ..pdf.extractor import ExtractedDocument
from ..reports.generator import extract_document, generate_report
from ..reports.models import Report
from ..utils.config import Limits
from ..utils.errors import AnalystError, LLMConfigError

log = logging.getLogger(__name__)


class BusyError(Exception):
    pass


class RateLimitedError(Exception):
    pass


@dataclass
class Job:
    id: str
    filename: str
    size_bytes: int
    created: float = field(default_factory=time.time)
    status: str = "queued"           # queued | reading | analyzing | done | error
    progress: float = 0.0
    message: str = "Waiting for a free worker"
    error: str | None = None
    error_is_config: bool = False
    document: ExtractedDocument | None = None
    report: Report | None = None
    cancelled: bool = False

    def document_info(self) -> dict | None:
        d = self.document
        if d is None:
            return None
        return {
            "filename": d.filename,
            "size_bytes": d.size_bytes,
            "pages": d.page_count,
            "text_pages": len(d.text_pages),
            "characters": d.total_chars,
            "characters_analysed": analysis_chars(d),
            "reference_pages": d.reference_pages,
            "ocr_pages": d.ocr_pages,
            "warnings": d.warnings,
        }

    def public(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "error": self.error,
            "error_is_config": self.error_is_config,
            "document": self.document_info(),
        }


class JobManager:
    def __init__(
        self,
        client_factory: Callable[[], object],
        limits_factory: Callable[[], Limits] = Limits.from_env,
        workers: int = 2,
        ttl_seconds: int = 7200,
        max_pending: int = 20,
        per_ip_per_hour: int = 30,
    ):
        self._client_factory = client_factory
        self._limits_factory = limits_factory
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_pending = max_pending
        self._per_ip = per_ip_per_hour
        self._ip_hits: dict[str, deque] = defaultdict(deque)

    # ------------------------------------------------------------------ public
    def submit(self, data: bytes, filename: str, ip: str = "-") -> Job:
        self._expire()
        with self._lock:
            hits = self._ip_hits[ip]
            now = time.time()
            while hits and now - hits[0] > 3600:
                hits.popleft()
            if self._per_ip and len(hits) >= self._per_ip:
                raise RateLimitedError()
            pending = sum(j.status in ("queued", "reading", "analyzing") for j in self._jobs.values())
            if pending >= self._max_pending:
                raise BusyError()
            hits.append(now)
            job = Job(id=uuid.uuid4().hex, filename=filename, size_bytes=len(data))
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, data)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job:
            job.cancelled = True      # a running analysis finishes but its result is discarded
        return job is not None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------------- internals
    def _expire(self) -> None:
        cutoff = time.time() - self._ttl
        with self._lock:
            for jid in [j.id for j in self._jobs.values() if j.created < cutoff]:
                self._jobs.pop(jid, None)

    def _set(self, job: Job, status: str | None = None, progress: float | None = None, message: str | None = None) -> None:
        if status:
            job.status = status
        if progress is not None:
            job.progress = max(0.0, min(1.0, progress))
        if message:
            job.message = message

    def _run(self, job: Job, data: bytes) -> None:
        if job.cancelled:
            return
        limits = self._limits_factory()
        try:
            self._set(job, "reading", 0.02, "Reading the PDF")
            job.document = extract_document(
                data, job.filename, limits,
                on_progress=lambda f, m: self._set(job, progress=0.02 + 0.13 * f, message=m),
            )
            del data
            if job.cancelled:
                return
            self._set(job, "analyzing", 0.15, "Starting the analysis")
            client = self._client_factory()
            try:
                report = generate_report(
                    job.document, client, limits,
                    lambda f, m: self._set(job, progress=0.15 + 0.85 * f, message=m),
                )
            finally:
                close = getattr(client, "close", None)
                if close:
                    close()
            if job.cancelled:
                return
            job.report = report
            self._set(job, "done", 1.0, "Done")
        except AnalystError as exc:
            log.info("job %s failed: %s | %s", job.id[:8], type(exc).__name__, exc.detail[:300])
            job.error = exc.user_message
            job.error_is_config = isinstance(exc, LLMConfigError)
            self._set(job, "error", message="Failed")
        except Exception:
            log.exception("job %s crashed", job.id[:8])
            job.error = "Something unexpected went wrong while generating the report. Please try again."
            self._set(job, "error", message="Failed")
