"""Stage-by-stage feasibility test for LoRA training on the course 27B INT4 checkpoint.

Each stage records success or the full error, and the report is rewritten after every stage,
so a failure shows exactly where the approach breaks. Nothing here is a real training run.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

USERDIR = Path(os.environ["TACC_USERDIR"])
MODEL = os.environ["QWEN_MODEL_PATH"]
OUT = USERDIR / "tacc-agentforge-results" / f"train-feasibility-{os.environ['SLURM_JOB_ID']}.json"
ADAPTER = USERDIR / "adapters" / f"feasibility-27b-{os.environ['SLURM_JOB_ID']}"
report = {"job": os.environ["SLURM_JOB_ID"], "model": MODEL, "stages": {}}
state = {}


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))


def stage(name):
    def wrap(fn):
        start = time.perf_counter()
        if any(not v["ok"] for v in report["stages"].values()):
            report["stages"][name] = {"ok": False, "skipped": "an earlier stage failed"}
            save()
            return False
        print(f"--- stage {name}", flush=True)
        try:
            result = fn()
            report["stages"][name] = {"ok": True, "seconds": round(time.perf_counter() - start, 1),
                                      "result": result}
        except BaseException as exc:  # includes CUDA OOM; report and stop the chain
            report["stages"][name] = {"ok": False, "seconds": round(time.perf_counter() - start, 1),
                                      "error": repr(exc)[:500],
                                      "trace": traceback.format_exc()[-3000:]}
            print(f"stage {name} FAILED: {exc!r}", flush=True)
            save()
            return False
        print(f"stage {name} ok: {json.dumps(report['stages'][name]['result'], default=str)[:400]}", flush=True)
        save()
        return True
    return wrap


def gpu_mem():
    import torch
    return [round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())]


@stage("imports")
def _():
    import torch
    import transformers
    info = {"env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_GPUS",
                                                  "GPU_DEVICE_ORDINAL", "NVIDIA_VISIBLE_DEVICES")},
            "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_total_gib": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
                              for i in range(torch.cuda.device_count())]}
    for mod in ["peft", "accelerate", "fla", "compressed_tensors", "causal_conv1d"]:
        try:
            info[mod] = getattr(__import__(mod), "__version__", "ok")
        except Exception as exc:
            info[mod] = f"MISSING: {exc!r}"[:150]
    return info


@stage("config")
def _():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    text = getattr(cfg, "text_config", cfg)
    quant = getattr(cfg, "quantization_config", None)
    return {"architectures": cfg.architectures, "model_type": cfg.model_type,
            "layers": getattr(text, "num_hidden_layers", None),
            "layer_types": sorted(set(getattr(text, "layer_types", []) or [])),
            "quantization": {k: quant.get(k) for k in ("quant_method", "format")} if isinstance(quant, dict) else str(quant)}


def load_model():
    import torch
    import transformers
    errors = {}
    # The checkpoint's keys are model.language_model.*, which only the image-text class expects;
    # AutoModelForCausalLM builds a text-only model and silently leaves the weights unloaded.
    for cls_name in ["AutoModelForImageTextToText", "AutoModelForCausalLM"]:
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto",
                                        max_memory={0: "21GiB", 1: "21GiB"})
            state["loader"] = cls_name
            return model
        except BaseException as exc:
            errors[cls_name] = repr(exc)[:400]
            torch.cuda.empty_cache()
    raise RuntimeError(f"all loaders failed: {errors}")


@stage("load_model")
def _():
    state["model"] = load_model()
    model = state["model"]
    return {"loader": state["loader"], "class": type(model).__name__,
            "gpu_alloc_gib": gpu_mem(),
            "param_dtypes": sorted({str(p.dtype) for p in model.parameters()})}


@stage("remove_decompress_hooks")
def _():
    """compressed-tensors attaches a forward hook that decompresses weights (and crashes on this
    checkpoint). We unpack the weights ourselves, so remove it."""
    removed = 0
    for mod in state["model"].modules():
        for hooks in (mod._forward_hooks, mod._forward_pre_hooks):
            for key, hook in list(hooks.items()):
                if "decompress" in repr(hook):
                    del hooks[key]
                    removed += 1
    return {"removed_hooks": removed}


@stage("inspect_modules")
def _():
    from collections import Counter
    model = state["model"]
    leaf_types = Counter()
    leaf_names = Counter()
    for name, mod in model.named_modules():
        if hasattr(mod, "weight") or hasattr(mod, "weight_packed"):
            leaf_types[type(mod).__name__] += 1
            leaf_names[name.split(".")[-1]] += 1
    state["leaf_names"] = leaf_names
    return {"module_types": dict(leaf_types.most_common(12)),
            "leaf_names": dict(leaf_names.most_common(25))}


def int4_dequant(mod):
    """Unpack a compressed-tensors 'pack-quantized' INT4 group-quantized Linear into a bf16 weight.

    Written from the format description because compressed_tensors 0.17's own dequantize fails
    on this checkpoint (it computes group_size 0). Layout: weight_packed [out, in/8] int32 holds 8
    unsigned 4-bit values per word (value + 8), element i at bits 4i; weight_zero_point is packed the
    same way along the output dimension ([out/8, groups]); weight_scale is [out, groups].
    """
    import torch
    packed, scale = mod.weight_packed, mod.weight_scale
    zp = getattr(mod, "weight_zero_point", None)
    # weight_shape is not reliably loaded (it held garbage), so derive the shape from the packed tensor.
    out_f, in_f = packed.shape[0], packed.shape[1] * 8
    groups = scale.shape[1]
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(out_f, in_f).to(torch.int16) - 8
    q = q.view(out_f, groups, in_f // groups)
    if zp is not None:
        z = ((zp.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).reshape(out_f, groups).to(torch.int16) - 8
        q = q - z.unsqueeze(-1)
    return (q.to(scale.dtype) * scale.unsqueeze(-1)).reshape(out_f, in_f).to(torch.bfloat16)


def quantized_forward(self, x):
    import torch
    with torch.no_grad():
        w = int4_dequant(self)
    return torch.nn.functional.linear(x, w.to(x.dtype), getattr(self, "bias", None))


@stage("dequant_check")
def _():
    """Show the stored tensors of one quantized layer and sanity-check the unpacked weight."""
    import torch
    model = state["model"]
    for name, mod in model.named_modules():
        if hasattr(mod, "weight_packed") and "language_model" in name and "mlp.gate_proj" in name:
            tensors = {n: [list(t.shape), str(t.dtype)] for n, t in
                       list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False))}
            w = int4_dequant(mod)
            return {"module": name, "tensors": tensors, "unpacked_shape": list(w.shape),
                    "mean": float(w.float().mean()), "std": float(w.float().std()),
                    "max_abs": float(w.float().abs().max()), "has_nan": bool(torch.isnan(w).any()),
                    "quantized_modules": sum(hasattr(m, "weight_packed") for m in model.modules())}
    raise RuntimeError("no module with weight_packed found")


@stage("forward_only")
def _():
    """Does the quantized model run in HuggingFace with our unpacking? (no grad, greedy decode)"""
    import types
    import torch
    from transformers import AutoTokenizer
    for mod in state["model"].modules():
        if hasattr(mod, "weight_packed"):
            mod.forward = types.MethodType(quantized_forward, mod)
    tok = AutoTokenizer.from_pretrained(MODEL)
    state["tok"] = tok
    model = state["model"]
    model.eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    return {"text": tok.decode(gen[0], skip_special_tokens=True), "seconds": round(time.perf_counter() - t0, 2),
            "gpu_alloc_gib": gpu_mem()}


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj")


def make_lora_class():
    import torch

    class LoRALinear(torch.nn.Module):
        """Adds a trainable low-rank update on top of a frozen (quantized) layer's own forward."""

        def __init__(self, base, r=8, alpha=16):
            super().__init__()
            self.base = base
            device = (list(base.parameters()) + list(base.buffers()))[0].device
            self.scale = alpha / r
            self.lora_A = torch.nn.Parameter(torch.empty(r, base.in_features, device=device, dtype=torch.float32))
            self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        def forward(self, x):
            out = self.base(x)
            update = (x.to(self.lora_A.dtype) @ self.lora_A.t()) @ self.lora_B.t()
            return out + (update * self.scale).to(out.dtype)

    return LoRALinear


@stage("lora")
def _():
    """peft cannot wrap these layers (no .weight), so wrap them with a small custom LoRA module."""
    import torch
    LoRALinear = make_lora_class()
    model = state["model"]
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if (child_name in LORA_TARGETS and isinstance(child, torch.nn.Linear)
                    and "language_model" in f"{name}.{child_name}"):
                setattr(mod, child_name, LoRALinear(child))
                replaced.append(f"{name}.{child_name}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"wrapped_modules": len(replaced), "examples": replaced[:3],
            "trainable_params": trainable, "gpu_alloc_gib": gpu_mem()}


@stage("train_steps")
def _():
    import torch
    from transformers import AutoTokenizer
    model = state["model"]
    tok = state.get("tok") or AutoTokenizer.from_pretrained(MODEL)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=1e-4)
    text = ("You are a shopping assistant. The user has a budget of 120 dollars. " * 40)
    results = []
    for seq_len in (256, 1024):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to("cuda:0")
        for i in range(3):
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)
            t0 = time.perf_counter()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            optim.step()
            optim.zero_grad()
            torch.cuda.synchronize()
            results.append({"seq_len": ids.shape[1], "step": i, "loss": round(out.loss.item(), 4),
                            "seconds": round(time.perf_counter() - t0, 2),
                            "peak_gib": [round(torch.cuda.max_memory_allocated(d) / 2**30, 2)
                                         for d in range(torch.cuda.device_count())]})
            print(results[-1], flush=True)
    return results


@stage("save_adapter")
def _():
    from safetensors.torch import save_file
    ADAPTER.mkdir(parents=True, exist_ok=True)
    tensors = {n: p.detach().cpu().contiguous() for n, p in state["model"].named_parameters()
               if p.requires_grad}
    save_file(tensors, str(ADAPTER / "lora_weights.safetensors"))
    files = {p.name: p.stat().st_size for p in ADAPTER.iterdir()}
    return {"path": str(ADAPTER), "tensors": len(tensors), "files": files, "sample_names": list(tensors)[:2]}


print("DONE", json.dumps({k: v["ok"] for k, v in report["stages"].items()}), flush=True)
