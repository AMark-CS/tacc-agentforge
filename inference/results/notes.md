## Findings

Each stage has 16 requests. The CUDA-graphs configuration ran three times in separate jobs (as `cuda-graphs`
in Group A, job 27784, and as `baseline` in Group B, jobs 27795 and 27798), which measures repeatability:
throughput and TPOT agree within about 1% (58.4 / 58.4 / 58.5 tok/s at c1; 203.9 / 203.5 / 203.4 at c8;
TPOT 14.9 ms in all three), while p50 TTFT varies more (distinct c4: 849 / 1091 / 1090 ms; shared-prefix
c8: 3110 / 2857 / 3465 ms, up to about 30%). So throughput and TPOT differences above roughly 5% are
meaningful, while TTFT differences need to be above roughly 30% to be trusted. Group B compares each variant
with the baseline from its own job. Group A's baseline is eager mode.

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

3. **Attention backend: FlashAttention-2 and Triton are equivalent.** In eager mode Triton is within about 5%
   on TPOT and throughput in every stage, and with CUDA graphs on it is identical (204.1 vs 203.4 tok/s at
   c8, TPOT 25.6 vs 25.8 ms, 59.2 vs 58.5 tok/s at c1). The one eager-mode TTFT outlier (+32% at c8) is inside
   the TTFT noise. FlashAttention-2 remains the default and the safer choice. Triton's KV cache was about
   10% smaller (174,307 vs 193,877 tokens) in that comparison; I did not investigate why.

4. **Disabling chunked prefill shows no consistent effect.** In eager mode throughput at c8 was +7% by the
   streaming client but -6% by `vllm bench serve`. With CUDA graphs on, throughput is identical to the
   baseline in every stage (203.8 vs 203.4 tok/s at c8). The only visible differences are at shared-prefix c4
   (TPOT 29.6 vs 40.6 ms, but TTFT 1892 vs 1286 ms), which I would not trust from one run given the noise
   above. The variant also changed `--max-num-batched-tokens` from 2048 to 4096 (vLLM requires it to be at
   least the 4096 context when chunking is off), so it is not a single-factor test, and vLLM warns that this
   model does not officially support disabling chunked prefill.

5. **Startup was slower on the first run.** It took 530 s, probably because the 19.6 GiB of weights
   had to be read from NFS (cold cache); later runs took 200-330 s. I did not isolate the cause.

6. **GPU memory.** Peak memory (NVML, job 27798) was 21.0 GiB per GPU for the CUDA-graphs baseline, 20.0 with
   Triton and 20.5 without chunked prefill, out of about 23.7 GiB. This is mostly memory vLLM reserves
   up front (85% target plus CUDA context and graphs), not what the model strictly needs: the weights take
   9.27 GiB per GPU.

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

- **Prefix caching plus the other flags, and kv-fp8.** Prefix caching was tested only on top of CUDA graphs and
  eager mode separately; combinations with Triton or chunked-prefill changes were not run.
- **Precision comparisons (INT8/INT4/FP8 weights).** Only the AWQ INT4 checkpoint is installed on the cluster, and
  no second checkpoint was downloaded, so weight-quantization differences are not measured. The compute
  nodes can reach Hugging Face and PyPI over HTTPS if Python is pointed at the system certificates
  (`SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`; HEAD requests returned HTTP 200 in job 27823). An
  earlier note here said the nodes had no internet access; that was wrong, the failure was Python not
  trusting the certificates.
- **PagedAttention.** It is always on in vLLM and cannot be disabled, and the block size is forced to
  784 tokens by the hybrid layers, so there was nothing to vary.
- **Repeats.** Only the CUDA-graphs baseline ran more than once; every other variant ran a single time.
- **Longer prompts and more concurrency.** Everything ran with a 4K context, 128 output tokens and up to 8
  concurrent requests.

## Next steps

1. Download a smaller second checkpoint (the nodes have network access, see above) and quantize it to
   FP8/INT8/INT4 for the weight-precision comparison. Installing a CUDA 12 toolchain for FlashInfer and FP8 KV
   from PyPI or conda is now possible in principle but untested.
2. Repeat `prefix-caching` at least twice more, and test a longer shared prefix and higher concurrency,
   since that is the variant with a real effect.
3. Recommended serving configuration from these runs: CUDA graphs on, prefix caching on for workloads with
   shared prompt prefixes (such as a fixed system prompt plus evidence), FlashAttention-2.
