import json
from pathlib import Path

from inference.report import load_runs, real_failure


def _write(folder: Path, variant: str, status: str, error: str = ""):
    folder.mkdir()
    stream = {"distinct-c8": {"requests": 2, "failed": 0}} if status == "ok" else {}
    (folder / "results.json").write_text(json.dumps(
        {"results": [{"variant": variant, "status": status, "error": error,
                      "streaming": stream}]}))


def test_retry_replaces_failed_attempt(tmp_path):
    _write(tmp_path / "a", "v", "failed", "boom")
    _write(tmp_path / "b", "v", "ok")
    runs = load_runs([tmp_path / "a", tmp_path / "b"])
    assert len(runs) == 1 and runs[0]["source"] == "b" and "boom" in runs[0]["retry_of"]


def test_dead_engine_marked_failed_even_if_status_ok():
    run = {"status": "ok", "streaming": {"s": {"requests": 4, "failed": 4}}}
    assert "every request failed" in real_failure(run)
