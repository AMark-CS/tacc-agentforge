#!/usr/bin/env bash
# Feasibility test: can the course 27B INT4 checkpoint take a LoRA training step on 2 GPUs?
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1
source /mnt/home/public/course-envs/CSIT6000V/project/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export QWEN_CACHE_HOME="$TACC_USERDIR/CSIT6000V-project-cache"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$TACC_USERDIR/pylib${PYTHONPATH:+:$PYTHONPATH}"
cd "$TACC_WORKDIR"
hostname
echo "before: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>} SLURM_GPUS=${SLURM_GPUS-<unset>}"
# Same as the verified serving script: Slurm's default hides the GPUs from this interpreter.
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
qwen-python -u main.py
