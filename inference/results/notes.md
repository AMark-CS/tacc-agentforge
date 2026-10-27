## Findings

Each stage has 16 requests and most variants ran once. The CUDA-graphs configuration happens to have run
twice (as `cuda-graphs` in Group A, job 27784, and as `baseline` in Group B, job 27795), which gives a
measure of repeatability: throughput and TPOT agree within about 1% (58.4 vs 58.4 tok/s at c1, 203.9 vs
203.5 at c8, TPOT 14.9 vs 14.9 ms), but p50 TTFT differs by up to about 30% (849 vs 1091 ms at c4 on
distinct prompts). So throughput/TPOT differences above roughly 5% are meaningful here, while TTFT
differences need to be above roughly 30% to be trusted. Group A's baseline is eager mode.

1. **CUDA graphs are the biggest single improvement.** TPOT at concurrency 1 fell from 86 ms to 15 ms
   (5.8x), and output throughput at concurrency 8 on distinct prompts rose from 67 to 204 tok/s
   (`vllm bench serve`: 68 to 166 tok/s). Startup is longer (300-330 s versus 215-230 s; `torch.compile`
   took 48-104 s) and the KV cache shrinks from 200,248 to about 174,000-194,000 tokens because graph
   capture reserves memory. TTFT does not improve: prefill is not launch-bound.

2. **Prefix caching helps a lot when requests share a prefix, and does nothing otherwise.** With CUDA graphs
   on (Group B), shared-prefix TTFT p50 at c8 fell from 2857 to 563 ms (-80%), throughput rose from 74 to
   232 tok/s (3.1x) and TPOT halved (48.8 to 24.1 ms); at c1 TTFT fell from 655 to 123 ms. In eager mode
   (Group A) the TTFT reduction was similar (about 70%) and throughput rose from 44 to 77 tok/s. On distinct
   prompts there is no effect in either group. Combined with CUDA graphs, shared-prefix throughput at c8 is
   about 5.3x the eager baseline (232 vs 44 tok/s). The cost is memory: the KV cache shrank by about a
   quarter in eager mode (200,248 to 150,186 tokens) and by about a third with CUDA graphs (193,877 to
   130,730 tokens), because vLLM switches this hybrid model's Mamba layers into an "align" cache mode that
   it labels experimental.

3. **Attention backend: FlashAttention-2 and Triton are equivalent here.** Triton is within about 5% on
   TPOT and throughput in every stage. Its TTFT at c8 was 32% higher on distinct prompts, which is inside
   the roughly 30% TTFT noise seen when the same configuration ran twice, so I do not count it as an
   effect. FlashAttention-2 is the default and the safer choice. This was tested in eager mode only.

4. **Disabling chunked prefill shows no consistent effect.** Throughput at c8 was +7% by the streaming
   client but -6% by `vllm bench serve`, and TTFT moved in both directions. This variant also changed
   `--max-num-batched-tokens` from 2048 to 4096 (vLLM requires it to be at least the 4096 context when
   chunking is off), so it is not a single-factor test. vLLM also warns that this model does not
   officially support disabling chunked prefill.

5. **Startup was slower on the first run.** It took 530 s, probably because the 19.6 GiB of weights
   had to be read from NFS (cold cache); later runs took 200-330 s. I did not isolate the cause.

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

- **Other flags under CUDA graphs.** Only prefix caching was repeated on the CUDA-graphs baseline. The
  attention-backend and chunked-prefill comparisons ran in eager mode only.
- **Precision comparisons (INT8/INT4/FP8 weights).** Only the AWQ INT4 checkpoint exists on the cluster
  and the nodes have no internet access to fetch another model. Weight-quantization differences
  are therefore not measured.
- **PagedAttention.** It is always on in vLLM and cannot be disabled, and the block size is forced to
  784 tokens by the hybrid layers, so there was nothing to vary.
- **Repeats.** Apart from the CUDA-graphs configuration noted above, no variant was run twice.

## Next steps

1. Run `attn-triton` and `no-chunked-prefill` on the CUDA-graphs baseline if those flags matter for the
   final setup, with at least 3 repeats.
2. Capture peak GPU memory: `nvidia-smi` was not found on the job's path, so only vLLM's own startup log
   figures (weights 9.27 GiB per GPU, KV cache sizes) are reported.
3. Decide with the course staff whether a CUDA 12 toolchain or a smaller second checkpoint can be made
   available, which would unlock FlashInfer, FP8 KV and the quantization comparison.
4. Recommended serving configuration from these runs: CUDA graphs on, prefix caching on for workloads with
   shared prompt prefixes (such as a fixed system prompt plus evidence), FlashAttention-2.
