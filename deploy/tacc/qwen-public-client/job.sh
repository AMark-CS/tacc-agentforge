#!/usr/bin/env bash
set -euo pipefail
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
cd "$TACC_WORKDIR"
export QWEN_BASE_URL="$(cat server-url.txt)"
case "$QWEN_BASE_URL" in *NODE*|'') echo 'Set server-url.txt to the running service address.' >&2; exit 1;; esac
qwen-python -u main.py
