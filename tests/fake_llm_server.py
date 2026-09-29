"""Local OpenAI-compatible HTTP server backed by tests/fake_llm.py.

Lets the real LLMClient / Streamlit app be exercised end to end without an API key or network.
For pipeline verification ONLY: the answers are mechanical extracts of the prompt, not analysis.

    python -m tests.fake_llm_server [port]      # then set LLM_BASE_URL=http://127.0.0.1:<port>/v1
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .fake_llm import fake_respond


class FakeLLMServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int = 0, hallucinate: bool = False, fail_first: int = 0, fail_status: int = 429):
        super().__init__(("127.0.0.1", port), _Handler)
        self.hallucinate = hallucinate
        self.fail_first = fail_first        # respond with fail_status to the first N requests
        self.fail_status = fail_status
        self.request_count = 0
        self.bodies: list[dict] = []
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def start(self) -> "FakeLLMServer":
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        srv: FakeLLMServer = self.server  # type: ignore[assignment]
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        with srv._lock:
            srv.request_count += 1
            n = srv.request_count
            srv.bodies.append(body)
        if n <= srv.fail_first:
            self._send(srv.fail_status, {"error": {"message": "simulated failure"}}, {"Retry-After": "0"})
            return
        msgs = body.get("messages", [])
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        try:
            content = fake_respond(system, user, srv.hallucinate)
        except Exception as exc:  # malformed / unknown task
            content = f"not json: {exc}"
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})

    def _send(self, status: int, payload: dict, headers: dict | None = None):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = FakeLLMServer(port)
    print(f"Fake LLM (test double) listening on {server.base_url}", flush=True)
    server.serve_forever()
