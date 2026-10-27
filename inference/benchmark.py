"""Streaming serving benchmark: TTFT, TPOT, throughput, end-to-end latency.

Standard library only so it runs on the TACC client and locally. Measurements here
are SSE-based and separate from the blocking client's total duration. Failures are
counted and kept in the denominator; they are never dropped.
"""
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.request import Request, urlopen


@dataclass
class RequestRecord:
    ok: bool
    error: str = ""
    ttft_s: float = 0.0          # request sent -> first content/reasoning chunk
    e2e_s: float = 0.0           # request sent -> stream end
    output_tokens: int = 0       # from usage if reported, else chunk count
    prompt_tokens: int = 0
    itl_s: list[float] = field(default_factory=list)  # gaps between chunks

    @property
    def tpot_s(self) -> float | None:
        """(e2e - TTFT) / (output_tokens - 1); undefined for <=1 output token."""
        if not self.ok or self.output_tokens <= 1:
            return None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)


@dataclass
class ServingRun:
    config_name: str
    n_requests: int
    n_failed: int
    wall_s: float
    records: list[RequestRecord]
    gpu_mem_gb: float = 0.0      # filled in by the experiment runner

    def summary(self) -> dict:
        ok = [r for r in self.records if r.ok]
        tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
        out_tokens = sum(r.output_tokens for r in ok)
        return {
            "config": self.config_name,
            "requests": self.n_requests,
            "failed": self.n_failed,
            "success_rate": len(ok) / self.n_requests if self.n_requests else 0.0,
            "wall_s": self.wall_s,
            "output_tokens": out_tokens,
            "output_tokens_per_s": out_tokens / self.wall_s if self.wall_s else 0.0,
            "requests_per_s": len(ok) / self.wall_s if self.wall_s else 0.0,
            "ttft_ms": _stats([r.ttft_s * 1000 for r in ok]),
            "tpot_ms": _stats([t * 1000 for t in tpot]),
            "itl_ms": _stats([g * 1000 for r in ok for g in r.itl_s]),
            "e2e_ms": _stats([r.e2e_s * 1000 for r in ok]),
            "gpu_mem_gb": self.gpu_mem_gb,
        }


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, 0 <= pct <= 100."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {"mean": statistics.fmean(values), "p50": percentile(values, 50),
            "p95": percentile(values, 95), "p99": percentile(values, 99)}


def stream_chat(base_url: str, model: str, prompt: str, max_tokens: int = 128,
                timeout_s: float = 300) -> RequestRecord:
    """One streaming chat request. base_url ends with /v1."""
    payload = {
        "model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(base_url.rstrip("/") + "/chat/completions",
                      data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    chunks, usage, gaps = 0, {}, []
    try:
        with urlopen(request, timeout=timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content") \
                            or delta.get("reasoning"):
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        else:
                            gaps.append(now - last)
                        last = now
                        chunks += 1
        end = time.perf_counter()
    except Exception as exc:  # network, HTTP, or JSON errors all count as failures
        return RequestRecord(ok=False, error=repr(exc), e2e_s=time.perf_counter() - start)
    if first is None:
        return RequestRecord(ok=False, error="no content received", e2e_s=end - start)
    return RequestRecord(ok=True, ttft_s=first - start, e2e_s=end - start,
                         output_tokens=usage.get("completion_tokens", chunks),
                         prompt_tokens=usage.get("prompt_tokens", 0), itl_s=gaps)


def run_benchmark(base_url: str, prompts: list[str], config_name: str, *,
                  model: str = "qwen3.8-27b", repeats: int = 3, concurrency: int = 1,
                  max_tokens: int = 128, warmup: int = 1) -> ServingRun:
    """Send every prompt `repeats` times with up to `concurrency` in flight."""
    for prompt in prompts[:warmup]:  # warmup is excluded from the results
        stream_chat(base_url, model, prompt, max_tokens=8)
    work = [p for _ in range(repeats) for p in prompts]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(
            lambda p: stream_chat(base_url, model, p, max_tokens=max_tokens), work))
    wall = time.perf_counter() - start
    return ServingRun(config_name, len(records), sum(not r.ok for r in records),
                      wall, records)
