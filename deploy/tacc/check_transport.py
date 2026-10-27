"""Task 0: verify SDK file round-trip and FTS5 on the CPU worker.

Run from a machine with the tcloud SDK configured. No model job is needed.
"""
import hashlib
import sqlite3
import uuid
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_probe_file(directory: Path) -> Path:
    """Write a uniquely named harmless file."""
    path = directory / f"probe-{uuid.uuid4().hex}.txt"
    path.write_text("agentforge transport probe\n", encoding="utf-8")
    return path


def roundtrip(local_file: Path) -> bool:
    """Upload, download, and compare hashes via the tcloud SDK."""
    raise NotImplementedError("TODO(Task 0): SDK upload/download")


def fts5_available() -> bool:
    """Run on the worker runtime: create and query an FTS5 table."""
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        con.execute("INSERT INTO t VALUES ('hello world')")
        return con.execute("SELECT count(*) FROM t WHERE t MATCH 'hello'").fetchone()[0] == 1
    except sqlite3.OperationalError:
        return False


if __name__ == "__main__":
    print("fts5:", fts5_available())
