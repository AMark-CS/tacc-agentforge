#!/usr/bin/env bash
# Usage: stage.sh <selftest|dryrun|collect> [max_calls]
# Copies the client, tests and query list next to the job files, then run `tcloud submit` here.
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:?selftest|dryrun|collect}; max_calls=${2:-50}
cp ../../../knowledge/serpapi.py ../../../tests/test_serpapi.py .
cp ../../../configs/shopping_queries.json .
printf '%s\n' "$mode" > mode.txt
printf '%s\n' "$max_calls" > max_calls.txt
sed "s/job-name=serpapi-[a-z]*/job-name=serpapi-$mode/" tuxiv.conf > tuxiv.conf.tmp && mv tuxiv.conf.tmp tuxiv.conf
echo "staged mode=$mode max_calls=$max_calls"
