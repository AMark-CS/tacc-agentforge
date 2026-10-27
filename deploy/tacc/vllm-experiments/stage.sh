#!/usr/bin/env bash
# Usage: [JOB_NAME=name] stage.sh <group> <gpus> [variant ...]
# Copies the runner into this directory and writes tuxiv.conf/group.txt. Then run
# `tcloud submit` from here. Staged copies are git-ignored.
set -euo pipefail
cd "$(dirname "$0")"
group=${1:?group name from configs/experiments.json}; gpus=${2:?GPUs per job}; shift 2
cp ../../../inference/benchmark.py ../../../inference/experiments.py .
cp ../../../configs/experiments.json .
# Descriptive Slurm name, e.g. exp-serving27b-prefix-no-chunked-kv-fp8 (max 40 chars).
jobname="exp-$group"; for v in "$@"; do jobname="$jobname-$v"; done
jobname=${JOB_NAME:-$jobname}
jobname=${jobname:0:40}
sed -e "s/__GPUS__/$gpus/" -e "s/__JOBNAME__/$jobname/g" tuxiv.conf.template > tuxiv.conf
printf '%s\n' "$group" > group.txt
if [ "$#" -gt 0 ]; then printf '%s\n' "$*" > only.txt; else rm -f only.txt; fi
echo "job-name=$jobname"
echo "staged group=$group gpus=$gpus only=${*:-all}"
