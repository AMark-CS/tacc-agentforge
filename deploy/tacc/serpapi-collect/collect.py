"""Collect Google Shopping and Google Product data with SerpApi, on TACC.

Reads the API key from $TACC_USERDIR/secrets/serpapi_key (the user places it there; it is never
printed or saved). Writes raw responses and an index under $TACC_USERDIR/data/serpapi/<run id>/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import serpapi as sp

USERDIR = Path(os.environ["TACC_USERDIR"])
parser = argparse.ArgumentParser()
parser.add_argument("--queries", type=Path, default=Path("shopping_queries.json"))
parser.add_argument("--key-file", type=Path, default=USERDIR / "secrets" / "serpapi_key")
parser.add_argument("--max-calls", type=int, default=50)
parser.add_argument("--dry-run", action="store_true", help="count planned calls; no key, no network")
args = parser.parse_args()

config = json.loads(args.queries.read_text(encoding="utf-8"))
queries, per_query = config["queries"], config.get("products_per_query", 5)
planned_max = len(queries) * (1 + per_query)
print(f"queries={len(queries)} products_per_query={per_query} worst_case_calls={planned_max} "
      f"max_calls={args.max_calls}", flush=True)
if args.dry_run:
    print("dry run: no calls made", flush=True)
    sys.exit(0)

key = sp.read_key(args.key_file)
run_dir = USERDIR / "data" / "serpapi" / f"run-{os.environ['SLURM_JOB_ID']}"
client = sp.SerpApiClient(key, run_dir, args.max_calls, locale=config.get("locale"))
summary = sp.collect(client, queries, per_query)
summary.update(run_dir=str(run_dir), job=os.environ["SLURM_JOB_ID"], max_calls=args.max_calls)
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("SUMMARY", json.dumps(summary, indent=2), flush=True)
