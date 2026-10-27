import json
from pathlib import Path

from inference.experiments import load_group, merge_args, server_command, workload_prompts

MATRIX = Path("configs/experiments.json")


def test_merge_args_changes_one_thing():
    base = ["--a", "1", "--eager", "--b", "2"]
    assert merge_args(base, {"--a": 9}, [], []) == ["--a", "9", "--eager", "--b", "2"]
    assert merge_args(base, {}, ["--eager"], []) == ["--a", "1", "--b", "2"]
    assert merge_args(base, {}, [], ["--c", "3"]) == base + ["--c", "3"]


def test_matrix_variants_differ_from_baseline_by_one_change():
    for group in json.loads(MATRIX.read_text())["groups"]:
        settings, variants = load_group(MATRIX, group)
        base = variants[0]
        for v in variants[1:]:
            assert v["args"] != base["args"] or v["env"] != base["env"], v["name"]
            assert v["name"] != "baseline"


def test_server_command_has_model_and_tp():
    settings, variants = load_group(MATRIX, "serving27b")
    cmd = server_command("python", "/m", settings, variants[0])
    assert cmd[cmd.index("--model") + 1] == "/m"
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"


def test_shared_prefix_prompts_share_a_prefix():
    prompts = workload_prompts("shared_prefix")
    assert len({p[:1000] for p in prompts}) == 1
    assert len(set(workload_prompts("distinct"))) == 8


def test_parse_server_log(tmp_path):
    from inference.experiments import parse_server_log
    log = tmp_path / "s.log"
    log.write_text("Model loading took 9.27 GiB memory\nAvailable KV cache memory: 10.55 GiB\n"
                   "GPU KV cache size: 200,248 tokens\nUsing FLASH_ATTN attention backend out of\n")
    facts = parse_server_log(log)
    assert facts["weights_gib"] == 9.27 and facts["kv_cache_tokens"] == 200248
    assert facts["attention_backend"] == "FLASH_ATTN"


def test_all_failures_mark_variant_failed(monkeypatch, tmp_path):
    import inference.experiments as ex
    from inference.benchmark import RequestRecord, ServingRun

    class FakeProc:
        pid = 0
        def poll(self): return None

    dead = ServingRun("v", 2, 2, 1.0, [RequestRecord(False, "boom"), RequestRecord(False, "boom")])
    monkeypatch.setattr(ex.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(ex, "wait_ready", lambda *a, **k: 1.0)
    monkeypatch.setattr(ex, "run_benchmark", lambda *a, **k: dead)
    monkeypatch.setattr(ex, "stop_server", lambda p: None)
    settings = {"served_name": "m", "tensor_parallel_size": 1, "startup_timeout_s": 1,
                "workload": {"input_len": 8, "output_len": 8, "num_prompts": 1,
                             "concurrency": [1], "prompt_kinds": ["distinct"], "repeats": 1}}
    rec = ex.run_variant("python", "/m", settings, {"name": "v", "env": {}, "args": []}, tmp_path)
    assert rec["status"] == "failed" and "every request failed" in rec["error"]


def test_gpu_memory_peak_tracks_per_gpu_and_total():
    import time
    from inference.experiments import GpuMemoryPeak
    readings = iter([[1.0, 2.0], [3.0, 1.0], [2.0, 2.0]])
    mem = GpuMemoryPeak(interval_s=0.01, reader=lambda: next(readings))
    mem.start()
    time.sleep(0.3)
    total = mem.stop()
    assert mem.peak_per_gpu_gb == [3.0, 2.0] and total == 4.0
