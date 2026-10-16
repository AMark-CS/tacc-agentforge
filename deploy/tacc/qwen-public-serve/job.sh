#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1
printf 'SDK_BOOTSTRAP=%s\n' "$CONDA_PREFIX"
source /mnt/home/public/course-envs/CSIT6000V/project/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export QWEN_CACHE_HOME="$TACC_USERDIR/CSIT6000V-project-cache"
mkdir -p -m 700 "$QWEN_CACHE_HOME"
hostname
printf 'SERVICE_BASE_URL=http://%s:8000/v1\n' "$(hostname)"
# Compatibility smoke test: same public model/runtime, eager execution and 4K context.
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
qwen-python -m vllm.entrypoints.openai.api_server \
    --model "$QWEN_MODEL_PATH" --served-model-name qwen3.8-27b \
    --safetensors-load-strategy prefetch --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 --language-model-only --dtype bfloat16 \
    --gpu-memory-utilization 0.85 --max-model-len 4096 \
    --max-num-seqs 2 --max-num-batched-tokens 1024 \
    --reasoning-parser qwen3 --enforce-eager --disable-custom-all-reduce
