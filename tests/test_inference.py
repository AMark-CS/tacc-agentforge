import pytest

from inference.client import ModelClient


@pytest.mark.skip(reason="Task 4: not implemented")
def test_chat_returns_result():
    ModelClient("http://localhost:8000/v1").chat([{"role": "user", "content": "hi"}])


def test_fts5_probe():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "check_transport", pathlib.Path("deploy/tacc/check_transport.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert isinstance(mod.fts5_available(), bool)


def _fake_server(chunks=4, fail=False):
    import json, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if fail:
                self.send_response(500); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for i in range(chunks):
                ev = {"choices": [{"delta": {"content": "x"}}]}
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
            usage = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": chunks}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def test_stream_chat_measures_tokens():
    from inference.benchmark import stream_chat
    server, url = _fake_server(chunks=4)
    rec = stream_chat(url, "m", "hi")
    server.shutdown()
    assert rec.ok and rec.output_tokens == 4 and rec.prompt_tokens == 5
    assert rec.ttft_s <= rec.e2e_s and rec.tpot_s is not None and len(rec.itl_s) == 3


def test_failures_stay_in_denominator():
    from inference.benchmark import run_benchmark
    server, url = _fake_server(fail=True)
    run = run_benchmark(url, ["a", "b"], "bad", model="m", repeats=2, warmup=0)
    server.shutdown()
    s = run.summary()
    assert s["requests"] == 4 and s["failed"] == 4 and s["success_rate"] == 0.0


def test_percentile_nearest_rank():
    from inference.benchmark import percentile
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([1, 2, 3, 4], 95) == 4
