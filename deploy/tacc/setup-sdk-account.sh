#!/usr/bin/env bash
# Configure the course with the original SDK; personal Conda is platform-provided.
set -euo pipefail
sdk=${TCLOUD_BIN:-tcloud}
task_root=/mnt/home/public/course-envs/CSIT6000V/assignment1
environments=$("$sdk" env ls 2>&1)
if [[ "$environments" == *'[Tcloud Error]'* || "$environments" != *'# conda environments:'* ]]; then
    printf '%s\nPlatform account initialization is incomplete; contact the TA.\n' "$environments" >&2
    exit 1
fi
listing=$("$sdk" ls -a -l ../ 2>&1)
[[ "$listing" != *'[Tcloud Error]'* ]]
has_config=false
while IFS= read -r entry; do
    if [[ "$entry" =~ (^|[[:space:]])\.condarc($|[[:space:]]) ]]; then
        has_config=true
    fi
done <<< "$listing"
if "$has_config"; then
    packages=$("$sdk" env ls -n mpi-gemm-v1 2>&1)
    if [[ "$packages" == *"# packages in environment at $task_root/mpi-gemm-v1:"* ]]; then
        echo 'Public course environment already configured.'
        exit 0
    fi
    echo 'Existing Conda configuration requires TA review; nothing changed.' >&2
    exit 1
fi
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/envs/mpi-gemm-v1"
printf 'envs_dirs:\n  - %s\n' "$task_root" > "$stage/.condarc"
# Destinations are relative to this account's USERDIR. Never use --cover.
"$sdk" upload "$stage/envs/mpi-gemm-v1" ../.Miniconda3/envs/ | tr -d '\000'
"$sdk" upload "$stage/.condarc" ../ | tr -d '\000'
packages=$("$sdk" env ls -n mpi-gemm-v1 2>&1)
[[ "$packages" == *"# packages in environment at $task_root/mpi-gemm-v1:"* ]]
echo 'Public course environment configured; no packages installed.'
