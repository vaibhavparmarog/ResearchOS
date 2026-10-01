"""In-memory analysis jobs. Nothing is written to disk and nothing is kept longer than needed.

Lifecycle of one upload:
  queued -> reading -> analyzing -> done | error | cancelled
- the PDF bytes are dropped as soon as the text is extracted;
- the extracted text is dropped as soon as the report exists;
- the report is handed to the browser once and purged shortly after (exports are generated
  from the browser's copy, see api.py);
- a job whose browser stops polling (tab closed) is cancelled after ABANDON_SECONDS, so it
  stops spending LLM quota;
- cancel() stops OCR, waits and LLM calls at the next checkpoint.
Job ids are random UUIDs; run the server with a single worker process.
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
from ..reports.generator import extract_document, generate_report
from ..reports.models import Report
from ..utils.config import Limits
from ..utils.errors import AnalystError, CancelledError, LLMConfigError

log = logging.getLogger(__name__)

ACTIVE = ("queued", "reading", "analyzing")


class BusyError(Exception):
    pass


class RateLimitedError(Exception):
    pass


class TooManyActiveError(Exception):
    pass


@dataclass
class Job:
    id: str
    filename: str
    size_bytes: int
    ip: str = "-"
    created: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    finished: float | None = None
    delivered: float | None = None
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting for a free worker"
    error: str | None = None
    error_is_config: bool = False
    doc_info: dict | None = None
    report: Report | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

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
            "document": self.doc_info,
        }


def _doc_info(d) -> dict:
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


class JobManager:
    def __init__(
        self,
        client_factory: Callable[..., object],
        limits_factory: Callable[[], Limits] = Limits.from_env,
        workers: int = 2,
        max_pending: int = 20,
        per_ip_per_hour: int = 30,
        max_active_per_ip: int = 2,
        abandon_seconds: float = 150.0,          # > background-tab timer throttling (~60 s)
        keep_finished_seconds: float = 120.0,
        keep_delivered_seconds: float = 30.0,
    ):
        self._client_factory = client_factory
        self._limits_factory = limits_factory
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="job")
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()          # queued job ids, for queue positions
        self._lock = threading.Lock()
        self._max_pending = max_pending
        self._per_ip = per_ip_per_hour
        self._max_active_per_ip = max_active_per_ip
        self._abandon = abandon_seconds
        self._keep_finished = keep_finished_seconds
        self._keep_delivered = keep_delivered_seconds
        self._ip_hits: dict[str, deque] = defaultdict(deque)
        self._stop = threading.Event()
        threading.Thread(target=self._reaper, daemon=True, name="job-reaper").start()

    # ------------------------------------------------------------------ public
    def submit(self, data: bytes, filename: str, ip: str = "-") -> Job:
        with self._lock:
            now = time.time()
            hits = self._ip_hits[ip]
            while hits and now - hits[0] > 3600:
                hits.popleft()
            if self._per_ip and len(hits) >= self._per_ip:
                raise RateLimitedError()
            active = [j for j in self._jobs.values() if j.status in ACTIVE]
            if self._max_active_per_ip and sum(j.ip == ip for j in active) >= self._max_active_per_ip:
                raise TooManyActiveError()
            if len(active) >= self._max_pending:
                raise BusyError()
            hits.append(now)
            job = Job(id=uuid.uuid4().hex, filename=filename, size_bytes=len(data), ip=ip)
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._pool.submit(self._run, job, data)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.last_seen = time.monotonic()
                if job.status == "queued" and job_id in self._order:
                    ahead = self._order.index(job_id)
                    job.message = "Waiting for a free worker" if ahead == 0 else f"In queue: {ahead} document(s) ahead"
            return job

    def mark_delivered(self, job: Job) -> None:
        with self._lock:
            if job.delivered is None:
                job.delivered = time.monotonic()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
        if job:
            job.cancel.set()
            if job.status in ACTIVE:
                job.status, job.message = "cancelled", "Cancelled"
            job.report = None
        return job is not None

    def active_count(self) -> int:
        with self._lock:
            return sum(j.status in ACTIVE for j in self._jobs.values())

    def shutdown(self) -> None:
        self._stop.set()
        for jid in list(self._jobs):
            self.cancel(jid)
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------------- internals
    def _reaper(self) -> None:
        while not self._stop.wait(5.0):
            now = time.monotonic()
            for job in list(self._jobs.values()):
                if job.status in ACTIVE and now - job.last_seen > self._abandon:
                    log.info("job %s abandoned by its browser; cancelling", job.id[:8])
                    self.cancel(job.id)
                elif job.delivered is not None and now - job.delivered > self._keep_delivered:
                    self.cancel(job.id)                       # purge the delivered report
                elif job.finished is not None and now - job.finished > self._keep_finished:
                    self.cancel(job.id)                       # never fetched: purge

    def _progress(self, job: Job, status: str | None = None, progress: float | None = None, message: str | None = None) -> None:
        if job.cancel.is_set():
            raise CancelledError()
        if status:
            job.status = status
        if progress is not None:
            job.progress = max(job.progress, min(1.0, progress))
        if message:
            job.message = message

    def _run(self, job: Job, data: bytes) -> None:
        with self._lock:
            if job.id in self._order:
                self._order.remove(job.id)
        if job.cancel.is_set():
            return
        limits = self._limits_factory()
        client = None
        try:
            self._progress(job, "reading", 0.02, "Reading the PDF")
            document = extract_document(
                data, job.filename, limits,
                on_progress=lambda f, m: self._progress(job, progress=0.02 + 0.13 * f, message=m),
            )
            data = b""                                   # drop the upload as soon as the text is out
            job.doc_info = _doc_info(document)
            self._progress(job, "analyzing", 0.15, "Starting the analysis")
            client = self._client_factory(job.cancel)
            if hasattr(client, "on_status"):
                client.on_status = lambda m: self._progress(job, message=m)   # live "what is happening" for the UI
            report = generate_report(
                document, client, limits,
                lambda f, m: self._progress(job, progress=0.15 + 0.85 * f, message=m),
            )
            del document                                 # the extracted text is not needed any more
            if job.cancel.is_set():
                return
            job.report = report
            job.status, job.progress, job.message = "done", 1.0, "Done"
        except CancelledError:
            job.status, job.message = "cancelled", "Cancelled"
        except AnalystError as exc:
            log.info("job %s failed: %s | %s", job.id[:8], type(exc).__name__, exc.detail[:300])
            job.error = exc.user_message
            job.error_is_config = isinstance(exc, LLMConfigError)
            job.status, job.message = "error", "Failed"
        except Exception:
            log.exception("job %s crashed", job.id[:8])
            job.error = "Something unexpected went wrong while generating the report. Please try again."
            job.status, job.message = "error", "Failed"
        finally:
            job.finished = time.monotonic()
            if client is not None and hasattr(client, "close"):
                client.close()
