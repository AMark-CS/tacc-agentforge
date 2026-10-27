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
