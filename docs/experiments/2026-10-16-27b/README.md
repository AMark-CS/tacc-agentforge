# 27B quantized service smoke test — 2026-10-16

## Outcome

The course-provided 27B INT4 checkpoint was served successfully through vLLM
on two allocated GPUs. A separate CPU client job called the OpenAI-compatible
chat endpoint and passed both checks. The service was explicitly stopped after
verification; the recorded node address is historical and is no longer live.

| Trial | Job | Node | Outcome |
| --- | --- | --- | --- |
| Original course compiled 32K service | 27410 | 10-0-1-10 | Weights loaded and rank-0 graph compilation completed, but API was not ready after more than 13 minutes; cancelled |
| Eager 4K compatibility service | 27411 | 10-0-1-11 | API startup complete; two requests answered; intentionally cancelled after testing |
| CPU client | 27412 | Separate CPU allocation | Both expected-answer checks passed; full JSON persisted |

## Verified configuration

- Public checkpoint label: `Qwen3.8-27B-AWQ-INT4`; API alias: `qwen3.8-27b`.
  This is the course-provided label, not an independently identified upstream release.
- Observed vLLM architecture: `Qwen3_5ForConditionalGeneration`; observed quantization
  backend: `compressed-tensors`, using `MarlinLinearKernel`.
- Public runtime: vLLM 0.26.0; prior GPU smoke test observed PyTorch 2.11.0+cu128.
- Tensor parallelism: 2; BF16 computation; eager execution; context limit 4096;
  maximum two concurrent sequences; maximum 1024 batched tokens.
- GPU memory utilization target: 0.85; prefix caching disabled in this test.
- NCCL P2P and IB paths disabled; custom all-reduce disabled. Several settings
  changed together, and the second trial used a different node, so this does
  not isolate the cause of the first trial's startup stall.
- No model downloads, shared environment edits, package installation, or new
  quantization were performed. Account setup configured public Conda lookup.

## Results

| Prompt | Expected | Actual | Completion tokens | Request duration |
| --- | --- | --- | --- | --- |
| 37加58等于多少？只输出数字。 | 95 | 95 | 3 | 14.880 s |
| 中国的首都是哪里？只输出城市名称。 | 北京 | 北京 | 2 | 0.304 s |

Temperature was zero, maximum output length 64 tokens, and thinking was disabled
through the chat template. The first request includes possible first-use
initialization overhead. Durations cover full HTTP requests, not TTFT, and two
tiny outputs cannot establish throughput, general quality, or agent reliability.

The successful server log reports 9.27 GiB for weights on each GPU, 0.12 GiB
peak activation, approximately 0.20 GiB non-Torch memory, and 10.55 GiB current
KV-cache memory. The 9.27 GiB figure is not total service VRAM consumption.

The first timestamped API log was at 08:45:39 UTC and API readiness was logged
at 08:49:04 UTC (16:45:39–16:49:04 Hong Kong time), a 3-minute-25-second interval.
This excludes job setup before the first timestamp and is not a promise for
subsequent startup latency.

Optional FlashInfer/CUDA_HOME warnings occurred during startup, but did not
prevent these requests from succeeding. Full 32K context, four-way concurrency,
streaming, tool calling, and an agent loop were not tested.

## Evidence and reproduction

- [Original startup log](serve-27410.log)
- [Successful service log, including shutdown](serve-27411.log)
- [Client test log](client-27412.log)
- [Full client responses and durations](client-27412.json)
- [SDK launch instructions](../../../deploy/tacc/README.md)

Next steps: add an agent tool loop; build a meaningful fixed evaluation set;
measure longer-output latency and throughput; compare against a same-model
unquantized baseline if resources permit. The earlier 8B trial was superseded
by the user's choice to validate the existing 27B service and was not executed.
