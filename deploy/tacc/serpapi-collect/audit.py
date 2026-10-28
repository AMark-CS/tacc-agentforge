"""Audit how complete the collected SerpApi data is (todo Task 1: audit representative records).

Reads one run directory (index.jsonl + raw/*.json), prints a summary and writes audit.json next to
it. Standard library only; nothing here calls the API or touches the key.
"""
import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path)
args = parser.parse_args()
root = Path(os.environ["TACC_USERDIR"]) / "data" / "serpapi"
run_dir = args.run_dir or sorted(root.glob("run-*"))[-1]

index = [json.loads(line) for line in (run_dir / "index.jsonl").read_text().splitlines()]
raw = {}
for rec in index:
    if rec.get("raw_path"):
        raw[rec["call_id"]] = json.loads((run_dir / rec["raw_path"]).read_text())


def frac(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def stats(values):
    return ({"min": min(values), "median": statistics.median(values), "max": max(values)}
            if values else None)


report = {"run_dir": str(run_dir), "calls": Counter(), "errors": Counter()}
for rec in index:
    report["calls"][f"{rec['engine']}:{rec['status']}"] += 1
    if rec["status"] != "ok":
        report["errors"][rec["error"][:80]] += 1

# ---- shopping results --------------------------------------------------------------------
shopping = [(rec, raw[rec["call_id"]]) for rec in index
            if rec["engine"] == "google_shopping" and rec["call_id"] in raw]
items = []
for rec, data in shopping:
    for item in data.get("shopping_results", []) or []:
        items.append({**item, "_query": rec["params"].get("q")})
unique = {i["product_id"]: i for i in items if i.get("product_id")}
fields = ["extracted_price", "rating", "reviews", "source", "delivery", "product_id",
          "immersive_product_page_token", "snippet", "old_price"]
report["shopping"] = {
    "queries": len(shopping),
    "result_rows": len(items),
    "unique_product_ids": len(unique),
    "results_per_query": {rec["params"]["q"]: len(data.get("shopping_results", []) or [])
                          for rec, data in shopping},
    "field_coverage": {f: frac(sum(1 for i in items if i.get(f) not in (None, "")), len(items)) for f in fields},
    "price_range_usd": stats([i["extracted_price"] for i in items if isinstance(i.get("extracted_price"), (int, float))]),
    "currency_symbols": dict(Counter((i.get("price") or "?")[:1] for i in items)),
}

# ---- product detail pages ----------------------------------------------------------------
products = [(rec, raw[rec["call_id"]]) for rec in index
            if rec["engine"] == "google_product" and rec["status"] == "ok" and rec["call_id"] in raw]
paths: dict[str, list] = defaultdict(list)


def walk(obj, path, depth):
    if depth > 3:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            p = f"{path}.{key}"
            paths[p].append(len(value) if isinstance(value, (list, dict, str)) else 1)
            walk(value, p, depth + 1)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for element in obj[:1]:
            walk(element, path + "[]", depth + 1)


for _, data in products:
    walk(data.get("product_results", {}), "product_results", 1)
census = {p: {"pages": len(v), "median_size": statistics.median(v)} for p, v in sorted(paths.items())}
review_keys = sorted(p for p in paths if "review" in p.lower())
store_counts = [len((d.get("product_results") or {}).get("stores") or []) for _, d in products]
attr_counts = [len((d.get("product_results") or {}).get("product_attributes") or []) for _, d in products]
store_items = [s for _, d in products for s in ((d.get("product_results") or {}).get("stores") or [])]
report["product_pages"] = {
    "ok_pages": len(products),
    "pages_with_stores": sum(1 for c in store_counts if c),
    "stores_per_page": stats(store_counts),
    "store_fields": {f: frac(sum(1 for s in store_items if s.get(f) not in (None, "")), len(store_items))
                     for f in sorted({k for s in store_items for k in s})},
    "pages_with_attributes": sum(1 for c in attr_counts if c),
    "attributes_per_page": stats(attr_counts),
    "pages_with_about_text": sum(1 for _, d in products if (d.get("product_results") or {}).get("about_the_product")),
    "pages_with_price_insights": sum(1 for _, d in products if (d.get("product_results") or {}).get("price_insights")),
    "review_related_keys": review_keys,
    "key_census": census,
}

# ---- can the failed detail lookups be retried with their page tokens? ---------------------
failed_ids = [rec["params"].get("product_id") for rec in index
              if rec["engine"] == "google_product" and rec["status"] != "ok"]
report["retry_with_token"] = {
    "failed_lookups": len(failed_ids),
    "have_page_token": sum(1 for pid in failed_ids if (unique.get(pid) or {}).get("immersive_product_page_token")),
}

(run_dir / "audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print("AUDIT", json.dumps({k: v for k, v in report.items() if k != "product_pages"}, indent=2, default=str))
pp = {k: v for k, v in report["product_pages"].items() if k != "key_census"}
print("PRODUCT_PAGES", json.dumps(pp, indent=2, default=str))
print("KEY_CENSUS (path: pages, median size)")
for p, v in census.items():
    print(f"  {p}: {v['pages']} pages, median size {v['median_size']}")
