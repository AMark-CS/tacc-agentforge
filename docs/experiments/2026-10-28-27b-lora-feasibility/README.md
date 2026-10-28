# LoRA fine-tuning feasibility on the course 27B INT4 model — 2026-10-28

## Outcome

A LoRA adapter can be trained on the course-provided `Qwen3.8-27B-AWQ-INT4` checkpoint on one TACC node
(2 GPUs, about 23.7 GiB each). The test ran 6 training steps and saved an adapter. It is a feasibility test,
not a training run: the loss values below come from one repeated sentence and say nothing about quality.

| Measurement | Value |
| --- | --- |
| Job | 27833 (`train-feasibility-27b`), node `10-0-1-11` |
| Model class | `Qwen3_5ForConditionalGeneration` (via `AutoModelForImageTextToText`) |
| Weights in GPU memory after load | 7.2 GiB + 11.6 GiB (INT4 stays packed) |
| LoRA | rank 8, alpha 16, 400 wrapped Linear layers in the language model, 54.4M trainable parameters |
| Step time | 3.1-3.3 s at 256 tokens; 4.6 s at 721 tokens (batch 1, gradient checkpointing) |
| Peak memory per GPU | 8.9 / 13.5 GiB at 256 tokens; 9.4 / 14.4 GiB at 721 tokens |
| First step | 140 s (probably kernel compilation; not isolated); model load 23 s |
| Correctness check | greedy completion of "The capital of France is" gave "Paris." for the unpacked INT4 weights |
| Saved adapter | `/mnt/home/markzhang/USERDIR/adapters/feasibility-27b-27833/lora_weights.safetensors` (217 MB, 800 tensors, fp32) |

Peak memory leaves about 9 GiB free on the busier GPU, so longer sequences or a batch larger than 1 should fit;
this was not tested. At about 5 s per 1k-token step, one one-hour job allows roughly 600-700 such steps
(an estimate, excluding the 2-3 minute startup).

## What it took (each item was a failure before it was a fix)

1. **Libraries.** `peft`, `accelerate` and `flash-linear-attention` were not installed. They were installed from
   PyPI into `/mnt/home/markzhang/USERDIR/pylib` (pure-Python wheels, `--no-deps`) using the Miniconda `pip`
   (Python 3.8, hence `--ignore-requires-python`). The jobs add that folder to `PYTHONPATH`; the course
   torch 2.11 and transformers 5.10.4 are untouched. See `deploy/tacc/train-env-setup/`.
2. **Network.** Compute nodes can reach PyPI and Hugging Face only with
   `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` (job 27823, `net-check-27823.json`).
3. **GPU visibility.** The job must export `CUDA_VISIBLE_DEVICES=0,1`, as the serving script does; without it
   PyTorch saw no GPU.
4. **Loader class.** `AutoModelForCausalLM` builds a text-only model whose layer names do not match the
   checkpoint (`model.language_model.*`), so the weights were silently left uninitialized (all-zero after
   unpacking). `AutoModelForImageTextToText` loads them correctly.
5. **Decompression hook.** `compressed-tensors` 0.17 attaches a forward hook that decompresses the weights
   and crashes on this checkpoint (`group_size` computed as 0). The test removes that hook and unpacks the
   weights itself (`int4_dequant` in `deploy/tacc/train-feasibility-27b/main.py`): INT4, group size 32,
   asymmetric, packed 8 values per int32, zero points packed along the output dimension. The stored
   `weight_shape` tensor is garbage after loading, so the shape is derived from the packed tensor.
6. **LoRA wrapper.** `peft` cannot wrap these layers (they have no `.weight`), so a small `LoRALinear`
   module adds the low-rank update on top of each frozen layer's own forward.

## Not verified

- **Serving the adapter.** The adapter is saved as a plain safetensors file with this test's own tensor names,
  not in `peft`'s format, so vLLM cannot load it yet. Writing `adapter_config.json` and `peft`-style names, and
  checking that vLLM 0.26 can apply a LoRA to this hybrid quantized model, are still open.
- **Training quality.** No real data, no held-out evaluation. The loss falling from 5.6 to 0.08 reflects
  repeated text.
- **Speed after warm-up for generation.** The 8-token generation took 136 s, dominated by first-call kernel
  compilation; steady-state generation speed was not measured.
- **`causal_conv1d`** is not installed, so the linear-attention layers use a fallback path.
- **Larger sequences and batches**, and a multi-job schedule (the one-hour job limit means checkpoint and resume).

## Files

- `train-feasibility-27833.json`: per-stage report of the successful run.
- `train-env-setup-27827.json`: library installation report.
- `net-check-27823.json`: network check.
- Job definitions: `deploy/tacc/train-feasibility-27b/`, `deploy/tacc/train-env-setup/`, `deploy/tacc/net-check/`.
