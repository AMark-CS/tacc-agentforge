"""Serving benchmark: TTFT, throughput, end-to-end latency, memory (todo Task 8).

Streaming measurements are separate from the blocking client's total duration.
Change one factor (batching, caching, parallelism, execution mode) per experiment.
"""
from dataclasses import dataclass


@dataclass
class ServingRun:
    config_name: str
    n_requests: int
    ttft_s: list[float]
    e2e_s: list[float]
    tokens_per_s: float
    gpu_mem_gb: float
    n_failed: int            # failures are counted, never dropped


def run_benchmark(base_url: str, prompts: list[str], config_name: str,
                  repeats: int = 3) -> ServingRun:
    raise NotImplementedError("TODO(Task 8): streaming load generator")
