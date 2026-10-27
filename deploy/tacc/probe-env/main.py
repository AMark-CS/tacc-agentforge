"""Inventory the TACC runtime: models, vLLM build, attention backends, network.

CPU-only. Writes one JSON report to USERDIR/tacc-agentforge-results/ and prints it.
"""
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


def run(cmd, timeout=60):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:  # report, never crash the probe
        return f"ERROR {exc!r}"


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def listing(path, limit=60):
    p = Path(path)
    if not p.is_dir():
        return None
    return sorted(x.name for x in p.iterdir())[:limit]


def reachable(url):
    try:
        urllib.request.urlopen(url, timeout=8)
        return True
    except Exception as exc:
        return f"no: {exc!r}"


packages = ["vllm", "torch", "transformers", "flash-attn", "flashinfer-python",
            "xformers", "bitsandbytes", "autoawq", "auto-gptq", "llmcompressor",
            "compressed-tensors", "peft", "trl", "datasets", "numpy", "aiohttp"]
report = {
    "job": os.environ.get("SLURM_JOB_ID"),
    "env": {k: v for k, v in os.environ.items() if k.startswith("QWEN_")},
    "packages": {p: version(p) for p in packages},
    "public_course_dir": listing("/mnt/home/public/course-envs/CSIT6000V"),
    "public_home": listing("/mnt/home/public"),
    "mnt_data": listing("/mnt/data"),
    "hf_cache_guess": listing(os.path.expanduser("~/.cache/huggingface/hub")),
    "model_path_listing": listing(os.environ.get("QWEN_MODEL_PATH", "/nonexistent")),
    "network": {"huggingface.co": reachable("https://huggingface.co"),
                "pypi.org": reachable("https://pypi.org")},
    "vllm_bench_serve_help": run(["qwen-python", "-m", "vllm.entrypoints.cli.main",
                                  "bench", "serve", "--help=all"])[:6000],
    "disk": run(["df", "-h", os.environ.get("TACC_USERDIR", "/")]),
    "which_qwen_python": shutil.which("qwen-python"),
}
out = Path(os.environ["TACC_USERDIR"]) / "tacc-agentforge-results"
out.mkdir(parents=True, exist_ok=True)
path = out / f"probe-env-{report['job']}.json"
path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print("REPORT", json.dumps(report, indent=2, ensure_ascii=False), flush=True)
print("Saved:", path, flush=True)
