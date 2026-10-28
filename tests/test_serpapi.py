"""Tests for the SerpApi client against a local fake server (no network, no real key).

Plain unittest so they run with `python -m unittest` on TACC, where pytest is not installed.
"""
import json
import os
import stat
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from knowledge import serpapi as sp
except ImportError:  # staged flat into the TACC job directory
    import serpapi as sp

KEY = "SECRET-TEST-KEY-123"


class FakeSerpApi:
    """Serves canned SerpApi-style JSON; `script` decides each response (status, body)."""

    def __init__(self, script):
        self.script, self.requests = script, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                outer.requests.append(query)
                status, body = outer.script(query, len(outer.requests))
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/search.json"

    def close(self):
        self.server.shutdown()


def normal(query, n):
    meta = {"id": f"search{n}", "processed_at": "2026-10-28 01:02:03 UTC", "total_time_taken": 1.5}
    if query["engine"] == "google_shopping":
        results = [{"product_id": str(100 + i), "title": f"Item {i}", "price": f"${10 + i}.99",
                    "source": "Shop", "immersive_product_page_token": f"tok{100 + i}",
                    "product_link": f"https://x/?api_key={KEY}"} for i in range(4)]
        return 200, {"search_metadata": meta, "shopping_results": results}
    return 200, {"search_metadata": meta, "product_results": {"title": "Detail"}}


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, fake, max_calls=50, **kw):
        return sp.SerpApiClient(KEY, self.out, max_calls, base_url=fake.url, sleep=lambda s: None, **kw)

    def test_key_never_saved(self):
        fake = FakeSerpApi(normal)
        try:
            client = self.client(fake)
            client.search_shopping("earbuds")
        finally:
            fake.close()
        for path in self.out.rglob("*"):
            if path.is_file():
                self.assertNotIn(KEY, path.read_text(), path)
        self.assertIn("REDACTED", next((self.out / "raw").iterdir()).read_text())
        self.assertEqual(fake.requests[0]["api_key"], KEY)  # but it is sent to the server

    def test_record_has_provenance(self):
        fake = FakeSerpApi(normal)
        try:
            record, data = self.client(fake).search_shopping("earbuds")
        finally:
            fake.close()
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.search_id, "search1")
        self.assertEqual(record.params["gl"], "us")
        self.assertNotIn("api_key", record.params)
        self.assertEqual(len(record.sha256), 64)
        line = json.loads((self.out / "index.jsonl").read_text().splitlines()[0])
        self.assertEqual(line["engine"], "google_shopping")

    def test_budget_is_a_hard_cap(self):
        fake = FakeSerpApi(normal)
        try:
            client = self.client(fake, max_calls=2)
            client.search_shopping("a")
            client.search_shopping("b")
            with self.assertRaises(sp.BudgetExceeded):
                client.search_shopping("c")
        finally:
            fake.close()
        self.assertEqual(client.calls_made, 2)
        self.assertEqual(len(fake.requests), 2)

    def test_retries_count_toward_budget(self):
        def flaky(query, n):
            return (500, {}) if n == 1 else normal(query, n)
        fake = FakeSerpApi(flaky)
        try:
            client = self.client(fake)
            record, data = client.search_shopping("a")
        finally:
            fake.close()
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.attempt, 2)
        self.assertEqual(client.calls_made, 2)

    def test_quota_exhaustion_stops(self):
        fake = FakeSerpApi(lambda q, n: (200, {"error": "Your account has run out of searches."}))
        try:
            client = self.client(fake)
            with self.assertRaises(sp.QuotaExhausted):
                client.search_shopping("a")
        finally:
            fake.close()
        self.assertEqual(json.loads((self.out / "index.jsonl").read_text().splitlines()[0])["status"], "error")

    def test_no_results_is_recorded_not_fatal(self):
        fake = FakeSerpApi(lambda q, n: (200, {"error": "Google hasn't returned any results for this query."}))
        try:
            record, data = self.client(fake).search_shopping("zzzz")
        finally:
            fake.close()
        self.assertEqual(record.status, "error")
        self.assertIn("hasn't returned", record.error)

    def test_bad_key_raises(self):
        fake = FakeSerpApi(lambda q, n: (401, {"error": "Invalid API key."}))
        try:
            with self.assertRaises(sp.AuthenticationFailed):
                self.client(fake).search_shopping("a")
        finally:
            fake.close()

    def test_collect_dedupes_and_respects_budget(self):
        fake = FakeSerpApi(normal)  # every query returns the same four product ids
        try:
            client = self.client(fake, max_calls=50)
            summary = sp.collect(client, [{"q": "a"}, {"q": "b"}], products_per_query=3)
        finally:
            fake.close()
        # 2 searches + 3 distinct products (the second query returns the same ids)
        self.assertEqual(summary["calls_made"], 5)
        self.assertEqual(summary["stopped"], "finished")

    def test_detail_lookup_uses_page_token(self):
        fake = FakeSerpApi(normal)
        try:
            client = self.client(fake)
            sp.collect(client, [{"q": "a"}], products_per_query=2)
        finally:
            fake.close()
        detail = [r for r in fake.requests if r["engine"] == "google_product"]
        self.assertEqual(len(detail), 2)
        self.assertTrue(all(r.get("page_token", "").startswith("tok") for r in detail))
        self.assertTrue(all("product_id" not in r for r in detail))

    def test_detail_lookup_falls_back_to_product_id(self):
        fake = FakeSerpApi(normal)
        try:
            self.client(fake).get_product(product_id="555")
            with self.assertRaises(ValueError):
                self.client(fake).get_product()
        finally:
            fake.close()
        self.assertEqual(fake.requests[0]["product_id"], "555")
        self.assertNotIn("page_token", fake.requests[0])

    def test_collect_stops_when_budget_runs_out(self):
        fake = FakeSerpApi(normal)
        try:
            client = self.client(fake, max_calls=3)
            summary = sp.collect(client, [{"q": "a"}, {"q": "b"}], products_per_query=5)
        finally:
            fake.close()
        self.assertEqual(client.calls_made, 3)
        self.assertTrue(summary["stopped"].startswith("budget"))

    def test_read_key_fixes_permissions(self):
        path = Path(self.tmp.name) / "key"
        path.write_text(KEY + "\n")
        os.chmod(path, 0o644)
        self.assertEqual(sp.read_key(path), KEY)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_read_key_rejects_missing_and_empty(self):
        with self.assertRaises(FileNotFoundError):
            sp.read_key(Path(self.tmp.name) / "nope")
        empty = Path(self.tmp.name) / "empty"
        empty.write_text("\n")
        with self.assertRaises(ValueError):
            sp.read_key(empty)

    def test_redact_removes_key_and_api_key_params(self):
        text = f"https://serpapi.com/x?api_key={KEY}&q=1 and {KEY}"
        out = sp.redact(text, KEY)
        self.assertNotIn(KEY, out)
        self.assertIn("api_key=REDACTED", out)


if __name__ == "__main__":
    unittest.main()
