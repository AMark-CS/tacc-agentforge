"""Run vLLM serving experiments inside one TACC job (todo Task 8).

For each variant in a group of configs/experiments.json: launch the OpenAI server
with that variant's flags, wait for readiness, run the streaming benchmark and
`vllm bench serve`, record peak GPU memory, then stop the server. One factor
changes per variant relative to the group's baseline. Failures (startup crash,
timeout, unsupported flag) are recorded in the results, never dropped.

Standard library only; `python` is the course interpreter (qwen-python on TACC).
Results are written after every variant so a killed job still leaves data.
"""
import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import urlopen

try:
    from inference.benchmark import run_benchmark
except ImportError:  # staged flat into the TACC job directory
    from benchmark import run_benchmark

PORT = 8000
STARTUP_MARGIN_S = 120  # stop launching variants this long before the deadline


def load_group(matrix_path: Path, group: str) -> tuple[dict, list[dict]]:
    """Return (group settings, variants) with the baseline's args merged in."""
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    settings = {**matrix.get("common", {}), **matrix["groups"][group]["settings"]}
    base = matrix["groups"][group]["baseline"]
    variants = [{"name": "baseline", "env": base.get("env", {}), "args": base["args"]}]
    for v in matrix["groups"][group]["variants"]:
        variants.append({"name": v["name"],
                         "env": {**base.get("env", {}), **v.get("env", {})},
                         "args": merge_args(base["args"], v.get("set", {}),
                                            v.get("remove", []), v.get("append", []))})
    return settings, variants


def merge_args(base: list[str], set_: dict, remove: list[str], append: list[str]) -> list[str]:
    """Replace `--flag value` pairs, drop flags, append new ones. One change per variant."""
    args, i, out = list(base), 0, []
    while i < len(args):
        flag = args[i]
        has_value = i + 1 < len(args) and not args[i + 1].startswith("--")
        if flag in remove:
            i += 2 if has_value else 1
            continue
        if flag in set_:
            out += [flag, str(set_[flag])]
            i += 2 if has_value else 1
            continue
        out += args[i:i + 2] if has_value else [flag]
        i += 2 if has_value else 1
    return out + append


def server_command(python: str, model_path: str, settings: dict, variant: dict) -> list[str]:
    return [python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path, "--served-model-name", settings["served_name"],
            "--host", "127.0.0.1", "--port", str(PORT),
            "--tensor-parallel-size", str(settings["tensor_parallel_size"]),
            *variant["args"]]


def bench_command(python: str, model_path: str, settings: dict, out_dir: Path,
                  name: str, concurrency: int) -> list[str]:
    w = settings["workload"]
    return [python, "-c", "from vllm.entrypoints.cli.main import main; main()",
            "bench", "serve", "--backend", "openai-chat",
            "--endpoint", "/v1/chat/completions",
            "--base-url", f"http://127.0.0.1:{PORT}",
            "--model", settings["served_name"], "--tokenizer", model_path,
            "--dataset-name", "random",
            "--random-input-len", str(w["input_len"]),
            "--random-output-len", str(w["output_len"]),
            "--num-prompts", str(w["num_prompts"]),
            "--max-concurrency", str(concurrency),
            "--save-result", "--result-dir", str(out_dir),
            "--result-filename", f"vllmbench-{name}-c{concurrency}.json"]


def workload_prompts(kind: str, n: int = 8) -> list[str]:
    """Deterministic prompts. `shared_prefix` exercises prefix caching."""
    prefix = ("You are a careful shopping assistant. Compare products using only the "
              "evidence below and cite it. " + "Evidence: item specs and reviews. " * 120)
    if kind == "shared_prefix":
        return [f"{prefix}\nQuestion {i}: which option is cheaper for budget {50 + i}?"
                for i in range(n)]
    return [f"Question {i}: " + "List tradeoffs of product option %d. " % i * 40
            for i in range(n)]


class GpuMemoryPeak(threading.Thread):
    """Poll nvidia-smi; keep the peak total used memory (GB) across visible GPUs."""

    def __init__(self, interval_s: float = 2.0):
        super().__init__(daemon=True)
        self.interval_s, self.peak_gb, self._stop_event = interval_s, 0.0, threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                self.peak_gb = max(self.peak_gb,
                                   sum(float(x) for x in out.split()) / 1024)
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self) -> float:
        self._stop_event.set()
        return self.peak_gb


def wait_ready(proc: subprocess.Popen, timeout_s: float) -> float | None:
    """Seconds until /health answers, or None on timeout / early exit."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if proc.poll() is not None:
            return None
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return time.perf_counter() - start
        except Exception:
            time.sleep(5)
    return None


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(15)  # let the GPU memory free before the next launch


def run_variant(python: str, model_path: str, settings: dict, variant: dict,
                out_dir: Path) -> dict:
    name = variant["name"]
    record = {"variant": name, "args": variant["args"], "env": variant["env"],
              "status": "failed", "error": ""}
    log = open(out_dir / f"server-{name}.log", "w")
    env = {**os.environ, **variant["env"]}
    mem = GpuMemoryPeak()
    mem.start()
    proc = subprocess.Popen(server_command(python, model_path, settings, variant),
                            stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    try:
        ready_s = wait_ready(proc, settings["startup_timeout_s"])
        if ready_s is None:
            record["error"] = "server not ready (crash or timeout); see server log"
            return record
        record["startup_s"] = ready_s
        workload = settings["workload"]
        record["streaming"] = {}
        for concurrency in workload["concurrency"]:
            for kind in workload["prompt_kinds"]:
                run = run_benchmark(
                    f"http://127.0.0.1:{PORT}/v1", workload_prompts(kind), name,
                    model=settings["served_name"], repeats=workload["repeats"],
                    concurrency=concurrency, max_tokens=workload["output_len"])
                record["streaming"][f"{kind}-c{concurrency}"] = run.summary()
        bench = subprocess.run(
            bench_command(python, model_path, settings, out_dir, name,
                          workload["concurrency"][-1]),
            capture_output=True, text=True, env=env, timeout=1200)
        record["vllm_bench"] = {"returncode": bench.returncode,
                                "tail": (bench.stdout + bench.stderr)[-3000:]}
        record["status"] = "ok"
    except Exception as exc:
        record["error"] = repr(exc)
    finally:
        record["peak_gpu_mem_gb"] = mem.stop()
        stop_server(proc)
        log.close()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiments.json"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=os.environ.get("VLLM_PYTHON", "qwen-python"))
    parser.add_argument("--deadline-s", type=float, default=3300,
                        help="stop launching variants when the job time is nearly up")
    parser.add_argument("--only", nargs="*", help="run just these variant names")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()

    settings, variants = load_group(args.matrix, args.group)
    model_path = os.path.expandvars(settings["model_path"])
    args.out.mkdir(parents=True, exist_ok=True)
    started, results = time.monotonic(), []
    for variant in variants:
        if args.only and variant["name"] not in args.only:
            continue
        if args.dry_run:
            print(variant["name"], variant["env"], " ".join(
                server_command(args.python, model_path, settings, variant)))
            continue
        remaining = args.deadline_s - (time.monotonic() - started)
        if remaining < settings["startup_timeout_s"] + STARTUP_MARGIN_S:
            results.append({"variant": variant["name"], "status": "skipped",
                            "error": f"only {remaining:.0f}s of job time left"})
            continue
        results.append(run_variant(args.python, model_path, settings, variant, args.out))
        (args.out / "results.json").write_text(
            json.dumps({"group": args.group, "settings": settings, "results": results},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print("DONE", variant["name"], results[-1]["status"], flush=True)


if __name__ == "__main__":
    main()
