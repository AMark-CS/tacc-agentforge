"""Validate the course model endpoint and persist complete smoke-test results."""
import json
import os
from pathlib import Path
import time
from qwen_client import chat


base_url = os.environ["QWEN_BASE_URL"]
tests = [("37加58等于多少？只输出数字。", "95"),
         ("中国的首都是哪里？只输出城市名称。", "北京")]
records = []
for prompt, expected in tests:
    started = time.perf_counter()
    result = chat([{"role": "user", "content": prompt}], base_url,
                  max_tokens=64, temperature=0)
    elapsed = time.perf_counter() - started
    answer = result["choices"][0]["message"]["content"]
    passed = isinstance(answer, str) and answer.strip().rstrip("。.!！") == expected
    records.append({"prompt": prompt, "expected": expected, "answer": answer,
                    "passed": passed, "elapsed_seconds": elapsed, "response": result})
    print("Question:", prompt, flush=True)
    print("Answer:", answer, flush=True)
    print("Usage:", result.get("usage", {}), flush=True)
    print("Elapsed seconds:", round(elapsed, 3), flush=True)
    print("PASS" if passed else "FAIL", flush=True)

output = Path(os.environ["TACC_USERDIR"]) / "tacc-agentforge-results"
output.mkdir(parents=True, exist_ok=True)
report = {"job_id": os.environ["SLURM_JOB_ID"], "base_url": base_url,
          "tests": records, "all_passed": all(record["passed"] for record in records)}
path = output / ("client-" + os.environ["SLURM_JOB_ID"] + ".json")
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print("Result saved:", path, flush=True)
if not report["all_passed"]:
    raise SystemExit("MODEL SMOKE TEST FAILED")
print("ALL MODEL SMOKE TESTS PASSED", flush=True)
