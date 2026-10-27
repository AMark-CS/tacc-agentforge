## Findings

All numbers come from single runs of 16 requests per stage, so only differences of roughly 20% or more
are treated as real. The baseline runs in eager mode, which matters for the comparisons below.

1. **CUDA graphs are the biggest single improvement.** Decode time per token (TPOT) at concurrency 1 fell
   from 86 ms to 15 ms (5.8x), and output throughput at concurrency 8 rose from 67 to 204 tok/s on
   distinct prompts (`vllm bench serve`: 68 to 166 tok/s). Startup cost is about 330 s versus 215-230 s,
   of which `torch.compile` took 104 s. KV cache shrank from 200,248 to 174,307 tokens because graph
   capture reserves memory. TTFT does not improve: prefill is not launch-bound, so p50 TTFT is roughly
   unchanged (it varies between 296 and 3110 ms across stages, with no consistent direction).

2. **Prefix caching helps a lot when requests share a prefix, and does nothing otherwise.**
   On the `shared_prefix` prompts TTFT p50 fell from 765 to 224 ms (c1), 1416 to 389 ms (c4) and
   2274 to 654 ms (c8), about 70% lower. Output throughput at c8 rose from 43.9 to 77.0 tok/s.
   On `distinct` prompts the results match the baseline. The price is memory: KV cache capacity dropped
   from 200,248 to 150,186 tokens, because vLLM switches this hybrid model's Mamba layers into an
   "align" cache mode that vLLM labels experimental.

3. **Attention backend: FlashAttention-2 and Triton are equivalent here.** Triton differs by under 5%
   in TPOT and throughput in every stage except one noisy TTFT reading at c8 (+32% on distinct prompts,
   +25% on shared-prefix), which does not appear at c1 or c4 and I would not call a real effect.
   FlashAttention-2 remains the default and the safer choice.

4. **Disabling chunked prefill shows no consistent effect.** Throughput at c8 was +7% by the streaming
   client but -6% by `vllm bench serve`, and TTFT moved in both directions. This variant also changed
   `--max-num-batched-tokens` from 2048 to 4096 (vLLM requires it to be at least the 4096 context when
   chunking is off), so it is not a single-factor test. vLLM also warns that this model does not
   officially support disabling chunked prefill.

5. **Startup was slower on the first run.** It took 530 s, probably because the 19.6 GiB of weights
   had to be read from NFS (cold cache); later runs took 200-230 s. I did not isolate the cause.

## Unsupported on this cluster

- **`attn-flashinfer`:** vLLM selected FlashInfer, but the engine died on the first request with
  `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`. FlashInfer compiles its
  kernels at run time. The shared vLLM environment contains `ninja`, CUDA headers and `ptxas`, but no
  `nvcc` binary, and the only `nvcc` on shared storage is CUDA 11.6, while this vLLM build uses CUDA 12.9.
  Fixing it needs a CUDA 12 toolkit and a C++ compiler on the nodes, for example through the course
  staff or an SDK conda environment. Neither was tried.
- **`kv-fp8`:** an FP8 KV cache cannot be used with FlashAttention-2, so vLLM chose FlashInfer, which
  failed for the same reason (every request returned HTTP 500). An untried alternative is to force the
  Triton attention backend (`--attention-backend TRITON_ATTN --kv-cache-dtype fp8`); the log lists it as
  an allowed backend for an FP8 cache, and Triton compiles with its own `ptxas`. Compare it with
  `attn-triton` to isolate the cache dtype.

## What was not tested

- **Combinations.** Each variant changes one flag from the baseline. CUDA graphs plus prefix caching,
  the obvious candidate for a production setup, was not run. The other comparisons were run in eager
  mode and may look different with CUDA graphs on.
- **Precision comparisons (INT8/INT4/FP8 weights).** Only the AWQ INT4 checkpoint exists on the cluster
  and the nodes have no internet access to fetch another model. Weight-quantization differences
  are therefore not measured.
- **PagedAttention.** It is always on in vLLM and cannot be disabled, and the block size is forced to
  784 tokens by the hybrid layers, so there was nothing to vary.
- **Repeatability.** Nothing was run twice, so node-to-node and run-to-run variation is not measured.

## Next steps

1. Run `prefix-caching` and `attn-triton` with CUDA graphs on, and the combination of both, against
   a CUDA-graph baseline with at least 3 repeats.
2. Capture peak GPU memory: `nvidia-smi` was not found on the job's path, so only vLLM's own startup log
   figures (weights 9.27 GiB per GPU, KV cache sizes) are reported.
3. Decide with the course staff whether a CUDA 12 toolchain or a smaller second checkpoint can be made
   available, which would unlock FlashInfer, FP8 KV and the quantization comparison.
