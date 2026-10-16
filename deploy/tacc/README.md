# Course-provided 27B AWQ service on TACC

These launch files originate from the public `qwen27b-student-release.zip`
provided by CSIT6000V. The shared runtime and checkpoint stay on TACC; no
model weights or credentials are included in this repository.

## Setup on the local client

Use the original TACC SDK. Configure your own username and private-key path,
run `tcloud init`, then run the provided account setup script:

```bash
export TCLOUD_BIN=/absolute/path/to/tcloud
bash setup-sdk-account.sh
```

The setup script configures this account's public Conda environment lookup.
The SDK's `mpi-gemm-v1` environment is a management bootstrap, not the LLM
runtime; job scripts require the public course prefix explicitly.

## Launch and test

```bash
cd qwen-public-serve
"$TCLOUD_BIN" submit
"$TCLOUD_BIN" cat slurm_log/qwen-public-serve-SERVICE_JOB_ID.log
```

Wait for `Application startup complete`, not merely a running job state.
Set `qwen-public-client/server-url.txt` to the `SERVICE_BASE_URL` printed in
the service log. This file is intentionally ignored by Git because the node
address changes between allocations.

```bash
cd ../qwen-public-client
"$TCLOUD_BIN" submit
"$TCLOUD_BIN" cat slurm_log/qwen-public-client-CLIENT_JOB_ID.log
```

The client checks arithmetic (`95`) and a Chinese factual answer (`北京`).
It stores complete JSON responses and request durations under
`TACC_USERDIR/tacc-agentforge-results/client-CLIENT_JOB_ID.json`.
These are smoke tests, not an accuracy benchmark or throughput evaluation.

Stop the service after testing:

```bash
"$TCLOUD_BIN" cancel -j SERVICE_JOB_ID
```

## Resource and runtime notes

- Service requests one node, two GPUs, eight CPUs, at most **one hour**.
  This differs from the upstream example's two-hour request because the
  observed `tacc` partition has a one-hour limit.
- Client requests one CPU, no GPU, at most ten minutes. Service and client
  occupy two simultaneous jobs, matching the observed per-user running-job limit.
- The service uses the public compatibility wrapper and checkpoint. After the
  original 32K compiled launch stalled, the local launcher uses eager execution,
  4K context, two concurrent requests, 85% GPU-memory utilization, and disables
  NCCL P2P/IB plus custom all-reduce. No shared files are modified.
- To reproduce the course's original compiled 32K configuration, replace the
  final launch command in `job.sh` with `bash "$QWEN_PROJECT_ROOT/serve.sh"`.
  This configuration was not verified successfully in the first local trial.
- Public checkpoint label: `Qwen3.8-27B-AWQ-INT4`; API model: `qwen3.8-27b`.
- The API is consumed by a client job inside TACC. Direct external HTTP access
  and SSH forwarding are not part of this workflow.
- Original course instructions: [STUDENT-COMMANDS.md](STUDENT-COMMANDS.md).
  Their two-hour resource example is superseded by the local one-hour configuration.

This milestone deploys a pre-quantized checkpoint; it does not perform new
quantization or implement an agent loop.
