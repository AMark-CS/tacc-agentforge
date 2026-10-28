#!/usr/bin/env bash
# CPU job. mode.txt is "selftest" (unit tests, no key, no network) or "collect" (real SerpApi calls).
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1
source /mnt/home/public/course-envs/CSIT6000V/project/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
mkdir -p -m 700 "$TACC_USERDIR/secrets" "$TACC_USERDIR/data/serpapi"
cd "$TACC_WORKDIR"
MODE="$(tr -d '[:space:]' < mode.txt)"
echo "mode=$MODE"
case "$MODE" in
  selftest) qwen-python -m unittest -v test_serpapi ;;
  collect)  qwen-python -u collect.py --max-calls "$(tr -d '[:space:]' < max_calls.txt)" ;;
  dryrun)   qwen-python -u collect.py --dry-run ;;
  audit)    qwen-python -u audit.py ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
