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
            if name == "baseline":
                runs.append(r)  # one baseline per job, so each variant has a same-job reference
            elif name not in index:
                index[name] = len(runs)
                runs.append(r)
            elif real_failure(runs[index[name]]) and not real_failure(r):
                r["retry_of"] = f"{runs[index[name]]['source']}: {real_failure(runs[index[name]])[:160]}"
                runs[index[name]] = r
    baselines = [r for r in runs if r["variant"] == "baseline"]
    for r in baselines:
        r["display"] = f"baseline ({r['source'].split('-')[-1]})" if len(baselines) > 1 else "baseline"
    return runs


def name(r: dict) -> str:
    return r.get("display", r["variant"])


def reference(runs: list[dict], r: dict) -> dict:
    """The baseline from the same job as `r`, else the first baseline."""
    baselines = [b for b in runs if b["variant"] == "baseline"]
    return next((b for b in baselines if b["source"] == r["source"]), baselines[0])


def mem_cell(r: dict) -> str:
    per_gpu = r.get("peak_gpu_mem_per_gpu_gb") or []
    return f"{max(per_gpu):.1f}" if per_gpu and max(per_gpu) > 0 else "n/a"


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


def summary_table(runs: list[dict], stage: str = "distinct-c8") -> str:
    lines = ["| Variant | Result | Startup (s) | KV cache (tokens) | GPU mem (GiB/GPU) | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        fail = real_failure(r)
        kv = r["facts"].get("kv_cache_tokens")
        if fail:
            lines.append(f"| {name(r)} | FAILED | {fmt(r.get('startup_s'), 0)} | "
                         f"{fmt(kv, 0)} | {mem_cell(r)} | - | - | - |")
            continue
        s = r["streaming"][stage]
        base = reference(runs, r)["streaming"][stage]
        lines.append(
            f"| {name(r)} | ok | {fmt(r.get('startup_s'), 0)} | {fmt(kv, 0)} | {mem_cell(r)} | "
            f"{fmt(s['output_tokens_per_s'])}{delta(s['output_tokens_per_s'], base['output_tokens_per_s'], False)} | "
            f"{fmt(s['ttft_ms']['p50'], 0)}{delta(s['ttft_ms']['p50'], base['ttft_ms']['p50'], True)} | "
            f"{fmt(s['tpot_ms']['p50'])}{delta(s['tpot_ms']['p50'], base['tpot_ms']['p50'], True)} |")
    return "\n".join(lines)


def detail_tables(runs: list[dict]) -> str:
    out = []
    for r in runs:
        if real_failure(r):
            continue
        out.append(f"#### {name(r)}\n")
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
            f"| {name(r)} | {b['completed']}/{b['failed']} | {fmt(b['output_throughput'])} | "
            f"{fmt(b['total_token_throughput'])} | {fmt(b['request_throughput'], 2)} | "
            f"{fmt(b['mean_ttft_ms'], 0)} / {fmt(b['p99_ttft_ms'], 0)} | "
            f"{fmt(b['mean_tpot_ms'])} / {fmt(b['p99_tpot_ms'])} | {fmt(b['mean_itl_ms'])} |")
    return "\n".join(lines)


def failure_list(runs: list[dict]) -> str:
    items = [f"- **{name(r)}**: {real_failure(r)[:400]}" for r in runs if real_failure(r)]
    return "\n".join(items) or "None."


GROUP_TITLES = {
    "serving27b": ("Group A: eager-mode baseline",
                   "Baseline runs with `--enforce-eager` (no CUDA graphs). Each variant changes one flag."),
    "graphs": ("Group B: CUDA-graphs baseline",
               "Baseline is the same configuration with CUDA graphs enabled (eager flag removed). "
               "Each variant changes one flag from this baseline."),
}


def group_section(group: str, runs: list[dict]) -> str:
    title, blurb = GROUP_TITLES.get(group, (f"Group {group}", ""))
    srcs = ", ".join(sorted({r["source"] for r in runs}))
    return f"""## {title}

{blurb} Source folders: {srcs}. Percentages compare each variant with the baseline from its own job;
changes under 3% are shown as "~same".

### Distinct prompts, concurrency 8

{summary_table(runs)}

### Shared-prefix prompts, concurrency 8

{summary_table(runs, "shared_prefix-c8")}

### Failed or unsupported variants

{failure_list(runs)}

### `vllm bench serve` (random dataset, concurrency 8)

{bench_table(runs)}

### Per-stage detail (streaming client)

{detail_tables(runs)}
"""


def build(folders: list[Path], notes: Path) -> str:
    by_group: dict[str, list[Path]] = {}
    for folder in folders:
        group = json.loads((folder / "results.json").read_text(encoding="utf-8")).get("group", folder.name)
        by_group.setdefault(group, []).append(folder)
    sections = "\n".join(group_section(g, load_runs(sorted(fs))) for g, fs in sorted(by_group.items(), key=lambda kv: list(GROUP_TITLES).index(kv[0]) if kv[0] in GROUP_TITLES else 99))
    text = f"""# vLLM serving experiments on TACC

Generated by `inference/report.py`. Raw data is in `inference/results/`.

## Setup

- **Model:** course-provided Qwen3.8-27B AWQ INT4 (compressed-tensors, Marlin kernels), served as `qwen3.8-27b`.
- **Runtime:** vLLM 0.26.0 (cu129), torch 2.11.0+cu128; 1 node, 2 GPUs, tensor parallel 2, 4K context.
  The GPU model is not recorded in the logs; the earlier test reports about 23.7 GiB per GPU, consistent with RTX 3090.
- **Common flags:** FlashAttention-2, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`, chunked prefill on,
  prefix caching off, `--gpu-memory-utilization 0.85`, `--disable-custom-all-reduce`. Group A adds
  `--enforce-eager`; Group B leaves CUDA graphs on.
- **Workload (streaming client, `inference/benchmark.py`):** 8 prompts x 2 repeats = 16 requests per stage,
  128 max output tokens, concurrency 1/4/8, two prompt sets (`distinct`, and `shared_prefix` with a
  common long prefix). One warm-up request is excluded. Failures stay in the denominator.
- **Workload (`vllm bench serve`):** random dataset, 512 input / 128 output tokens, 32 prompts, concurrency 8.
- **Metrics:** TTFT = request sent to first streamed token. TPOT = (end-to-end - TTFT) / (output tokens - 1).
  Output tok/s = total output tokens / wall time of the stage.

{sections}
## Limitations

- Each stage has 16 requests and one repeat set, so differences of a few percent are within noise.
- Variants ran in different jobs and possibly on different nodes. The only repeat is the CUDA-graphs
  configuration, which ran twice (`cuda-graphs` in Group A's job and `baseline` in Group B); see Findings
  for how far those two agree.
- `shared_prefix` and `distinct` prompts produce different output lengths, so compare TTFT and TPOT
  across prompt sets rather than raw tokens/s.
- Peak GPU memory was captured (through NVML) only in job 27798; earlier jobs show "n/a" because
  nvidia-smi was not reachable. It reflects memory vLLM reserves (85% utilization target), not the minimum
  the model needs. Weight and KV-cache sizes from vLLM's startup log are shown for every run.
- PagedAttention is always on in vLLM and cannot be disabled, so it is not an on/off variant here.
  Block size cannot be varied on this model because vLLM forces 784-token blocks for the hybrid
  (Mamba + attention) layers. Quantization precision comparisons need a second checkpoint, which was not
  downloaded.
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
