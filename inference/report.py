"""Turn experiment result folders into one markdown report (inference/RESULTS.md).

Usage:
    python inference/report.py inference/results/serving27b-* --out inference/RESULTS.md

Each folder holds the results.json written by inference/experiments.py, the
vllmbench-<variant>-c<N>.json files from `vllm bench serve`, and optionally
facts-<variant>.txt (lines grepped from the server log) for older runs. Hand-written
analysis lives in inference/results/notes.md and is appended if present.
"""
import argparse
import json
from pathlib import Path

try:
    from inference.experiments import parse_server_log
except ImportError:
    from experiments import parse_server_log

STAGES = ["distinct-c1", "distinct-c4", "distinct-c8",
          "shared_prefix-c1", "shared_prefix-c4", "shared_prefix-c8"]


def load_runs(folders: list[Path]) -> list[dict]:
    """One dict per variant, tagged with its source folder.

    The first folder's result wins, except that a later successful run replaces an
    earlier failed one (a retried variant); `retry_of` then names the failed attempt.
    """
    runs, index = [], {}
    for folder in folders:
        data = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        for r in data["results"]:
            if r["status"] == "skipped":
                continue
            r["source"] = folder.name
            facts = r.get("server_log_facts") or {}
            facts_file = folder / f"facts-{r['variant']}.txt"
            if not facts and facts_file.exists():
                facts = parse_server_log(facts_file)
            r["facts"] = facts
            bench = folder / f"vllmbench-{r['variant']}-c8.json"
            r["bench"] = json.loads(bench.read_text()) if bench.exists() else None
            name = r["variant"]
            if name not in index:
                index[name] = len(runs)
                runs.append(r)
            elif real_failure(runs[index[name]]) and not real_failure(r):
                r["retry_of"] = f"{runs[index[name]]['source']}: {real_failure(runs[index[name]])[:160]}"
                runs[index[name]] = r
    return runs


def real_failure(r: dict) -> str:
    """Status in the file can be stale (older runs marked a dead engine 'ok')."""
    if r["status"] != "ok":
        return r.get("error") or r["status"]
    stream = r.get("streaming", {})
    if stream and all(s["failed"] == s["requests"] for s in stream.values()):
        return "every request failed (engine died after startup)"
    return ""


def fmt(x, digits=1, suffix=""):
    return "n/a" if x is None else f"{x:.{digits}f}{suffix}"


def delta(value, base, lower_is_better):
    if value is None or not base or value == base:
        return ""
    pct = (value - base) / base * 100
    good = pct < 0 if lower_is_better else pct > 0
    return f" ({pct:+.0f}% {'better' if good else 'worse'})" if abs(pct) >= 3 else " (~same)"


def summary_table(runs: list[dict]) -> str:
    base = next(r for r in runs if r["variant"] == "baseline")["streaming"]["distinct-c8"]
    lines = ["| Variant | Result | Startup (s) | KV cache (tokens) | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        fail = real_failure(r)
        facts = r["facts"]
        kv = facts.get("kv_cache_tokens")
        if fail:
            lines.append(f"| {r['variant']} | FAILED | {fmt(r.get('startup_s'), 0)} | "
                         f"{fmt(kv, 0)} | - | - | - |")
            continue
        s = r["streaming"]["distinct-c8"]
        lines.append(
            f"| {r['variant']} | ok | {fmt(r.get('startup_s'), 0)} | {fmt(kv, 0)} | "
            f"{fmt(s['output_tokens_per_s'])}{delta(s['output_tokens_per_s'], base['output_tokens_per_s'], False)} | "
            f"{fmt(s['ttft_ms']['p50'], 0)}{delta(s['ttft_ms']['p50'], base['ttft_ms']['p50'], True)} | "
            f"{fmt(s['tpot_ms']['p50'])}{delta(s['tpot_ms']['p50'], base['tpot_ms']['p50'], True)} |")
    return "\n".join(lines)


def detail_tables(runs: list[dict]) -> str:
    out = []
    for r in runs:
        if real_failure(r):
            continue
        out.append(f"#### {r['variant']}\n")
        out.append("| Stage | OK/total | Output tok/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | E2E p50 (ms) |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for st in STAGES:
            s = r["streaming"].get(st)
            if not s:
                continue
            out.append(f"| {st} | {s['requests'] - s['failed']}/{s['requests']} | "
                       f"{fmt(s['output_tokens_per_s'])} | "
                       f"{fmt(s['ttft_ms']['p50'], 0)} / {fmt(s['ttft_ms']['p95'], 0)} | "
                       f"{fmt(s['tpot_ms']['p50'])} / {fmt(s['tpot_ms']['p95'])} | "
                       f"{fmt(s['e2e_ms']['p50'], 0)} |")
        out.append("")
    return "\n".join(out)


def bench_table(runs: list[dict]) -> str:
    lines = ["| Variant | Completed/failed | Output tok/s | Total tok/s | Req/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL mean (ms) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        b = r["bench"]
        if not b or real_failure(r):
            continue
        lines.append(
            f"| {r['variant']} | {b['completed']}/{b['failed']} | {fmt(b['output_throughput'])} | "
            f"{fmt(b['total_token_throughput'])} | {fmt(b['request_throughput'], 2)} | "
            f"{fmt(b['mean_ttft_ms'], 0)} / {fmt(b['p99_ttft_ms'], 0)} | "
            f"{fmt(b['mean_tpot_ms'])} / {fmt(b['p99_tpot_ms'])} | {fmt(b['mean_itl_ms'])} |")
    return "\n".join(lines)


def failure_list(runs: list[dict]) -> str:
    items = [f"- **{r['variant']}**: {real_failure(r)[:400]}" for r in runs if real_failure(r)]
    return "\n".join(items) or "None."


def build(folders: list[Path], notes: Path) -> str:
    runs = load_runs(folders)
    sources = ", ".join(sorted({r["source"] for r in runs}))
    text = f"""# vLLM serving experiments on TACC

Generated by `inference/report.py` from: {sources}. Raw data is in `inference/results/`.

## Setup

- **Model:** course-provided Qwen3.8-27B AWQ INT4 (compressed-tensors, Marlin kernels), served as `qwen3.8-27b`.
- **Runtime:** vLLM 0.26.0 (cu129), torch 2.11.0+cu128; 1 node, 2 GPUs, tensor parallel 2, 4K context.
  The GPU model is not recorded in the logs; the earlier test reports about 23.7 GiB per GPU, consistent with RTX 3090.
- **Baseline flags:** eager execution, FlashAttention-2, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`,
  chunked prefill on, prefix caching off, `--gpu-memory-utilization 0.85`, `--disable-custom-all-reduce`.
  Each variant changes one thing relative to this baseline.
- **Workload (streaming client, `inference/benchmark.py`):** 8 prompts x 2 repeats = 16 requests per stage,
  128 max output tokens, concurrency 1/4/8, two prompt sets (`distinct`, and `shared_prefix` with a
  common long prefix). One warm-up request is excluded. Failures stay in the denominator.
- **Workload (`vllm bench serve`):** random dataset, 512 input / 128 output tokens, 32 prompts, concurrency 8.
- **Metrics:** TTFT = request sent to first streamed token. TPOT = (end-to-end - TTFT) / (output tokens - 1).
  Output tok/s = total output tokens / wall time of the stage.

## Summary (distinct prompts, concurrency 8)

Percentages compare with the baseline; changes under 3% are shown as "~same".

{summary_table(runs)}

## Failed or unsupported variants

{failure_list(runs)}

## `vllm bench serve` (random dataset, concurrency 8)

{bench_table(runs)}

## Per-stage detail (streaming client)

{detail_tables(runs)}
## Limitations

- Each stage has 16 requests and one repeat set, so differences of a few percent are within noise.
- Variants ran in different jobs and possibly on different nodes; no run was repeated, so node-to-node
  variation is not separated from the effect of the flag.
- `shared_prefix` and `distinct` prompts produce different output lengths, so compare TTFT and TPOT
  across prompt sets rather than raw tokens/s.
- Peak GPU memory was not captured in these runs (nvidia-smi was not reachable); memory figures come from
  vLLM's own startup log (weights and KV-cache size).
- PagedAttention is always on in vLLM and cannot be disabled, so it is not an on/off variant here.
  Block size cannot be varied on this model because vLLM forces 784-token blocks for the hybrid
  (Mamba + attention) layers. Quantization precision comparisons need a second checkpoint, which is not
  available on the cluster.
"""
    if notes.exists():
        text += "\n" + notes.read_text(encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("inference/RESULTS.md"))
    parser.add_argument("--notes", type=Path, default=Path("inference/results/notes.md"))
    args = parser.parse_args()
    args.out.write_text(build(sorted(args.folders), args.notes), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
