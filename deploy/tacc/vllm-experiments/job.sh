#!/usr/bin/env bash
# Run one experiment group (named in group.txt) inside a single TACC job.
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
mkdir -p -m 700 "$QWEN_CACHE_HOME"
cd "$TACC_WORKDIR"
GROUP="$(tr -d '[:space:]' < group.txt)"
ONLY="$(cat only.txt 2>/dev/null || true)"   # optional: space-separated variant names
OUT="$TACC_USERDIR/tacc-agentforge-results/exp-$GROUP-$SLURM_JOB_ID"
hostname
# Time limit is 1h; leave room to flush results.
qwen-python -u experiments.py --matrix experiments.json --group "$GROUP" \
    --out "$OUT" --deadline-s 3300 ${ONLY:+--only $ONLY}
echo "RESULTS_DIR=$OUT"
