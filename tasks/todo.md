# Shopping advisor implementation tasks

All tasks below are planned. Dependencies refer to task numbers in this file.

## Task 0: Verify TACC file transport

Confirm the native Windows SDK can upload and download request/result files and
that the CPU worker has SQLite FTS5. Reuse the current deployment entrypoints.

- [ ] A uniquely named harmless file completes a round-trip with an identical hash.
- [ ] The selected worker runtime can create/query an FTS5 table.
- [ ] Request IDs, persistent USERDIR paths, and deadline behavior are documented.

**Verification:** SDK round-trip and worker capability smoke checks; no model job
is necessary for the file-transfer check.
**Dependencies:** None. **Scope:** Small.
**Likely files:** `deploy/tacc/README.md`, `deploy/tacc/qwen-public-client/job.sh`.

## Task 1: Collect auditable SerpApi records

Add the backend acquisition adapter for Shopping discovery and Product enrichment.
Keep credentials in the backend environment and fixtures free of secrets.

- [ ] A small live sample is archived with query/locale, source IDs, and timestamps.
- [ ] Missing fields, unavailable reviews, quota failures, and timeouts are explicit.
- [ ] Retry/call limits and secret redaction pass focused fixture tests.

**Verification:** Adapter tests using sanitized success/error fixtures; a small
quota-bounded live acquisition after an API key is configured.
**Dependencies:** None. **Scope:** Medium.
**Likely files:** `knowledge/serpapi.py`, `tests/test_serpapi.py`,
`tests/fixtures/serpapi.json`, `.env.example`.

## Task 2: Build versioned knowledge snapshots

Normalize the acquisition output into products, variants, offers, and evidence;
attach provenance and provide FTS5 retrieval.

- [ ] Variant boundaries, decimal costs, currencies, missing fields, and duplicates are preserved correctly.
- [ ] Retrieval returns ranked evidence IDs and resolvable source links.
- [ ] A manifest exports/reloads a frozen snapshot with matching hashes and counts.

**Verification:** Normalization, retrieval, deduplication, and snapshot round-trip
tests; manual coverage audit of the collected sample.
**Dependencies:** 1. **Scope:** Medium.
**Likely files:** `knowledge/store.py`, `knowledge/schema.sql`,
`tests/test_knowledge.py`, `tests/fixtures/products.json`.

## Checkpoint after collection and indexing

- [ ] The raw-record-to-cited-evidence path works on the sample.
- [ ] Missing review coverage and freshness limits are documented.
- [ ] Secret scanning finds no API keys or private credentials in fixtures/results.

## Task 3: Implement exact budget validation

Validate quantities, essential requirements, known delivery/taxes, purchase caps,
periodic balances, spending, and reserves without delegating arithmetic to the LLM.

- [ ] Exact arithmetic gives correct itemized totals and remaining funds at boundaries.
- [ ] Unknown essential fees, incompatible units/currencies, or absent required attributes trigger clarification.
- [ ] The selected plan cites its price/requirement evidence from the chosen snapshot.

**Verification:** Focused tests for rounding, zero/negative quantities, insufficient
funds, missing fees, and the smaller-of-two-budgets rule.
**Dependencies:** 2. **Scope:** Small.
**Likely files:** `agent/budget.py`, `tests/test_budget.py`.

## Task 4: Complete the bounded agent loop

Reuse the model client for clarification, evidence retrieval, comparison, budget
validation, and structured recommendations.

- [ ] A complete shopping request returns a cited valid plan or a justified clarification.
- [ ] Tool schemas, allowed actions, maximum steps, and deadlines bound execution.
- [ ] Missing-evidence requests, invalid model outputs, and instruction-like evidence fail predictably.

**Verification:** Mock-model traces for success, clarification, tool failure,
step exhaustion, and evidence injection; one TACC-backed request after task 0.
**Dependencies:** 0, 2, 3. **Scope:** Medium.
**Likely files:** `agent/runtime.py`, `agent/tools.py`,
`deploy/tacc/qwen-public-client/qwen_client.py`, `tests/test_agent.py`.

## Task 5: Connect the browser and worker

Add the backend request/result bridge and a minimal browser interface for input,
progress, sources, ranked plans, and revisions.

- [ ] Request IDs map browser sessions, backend tool requests, snapshots, and worker results without mixing sessions.
- [ ] Public API calls stay on the backend; model HTTP calls stay inside TACC.
- [ ] One complete request plus revision works, including expired-job and retry states.

**Verification:** Bridge round-trip tests, backend contract tests, and a browser
check covering progress, source inspection, revision, and timeout recovery.
**Dependencies:** 0, 4. **Scope:** Medium; split backend/transport and UI into
successive implementation slices if more than five files are needed.
**Likely files:** `app/backend.py`, `app/bridge.py`, `app/index.html`,
`deploy/tacc/qwen-public-client/main.py`, `tests/test_bridge.py`.

## Checkpoint after the complete user request

- [ ] A browser request returns a source-linked, budget-checked recommendation.
- [ ] Evidence IDs resolve to archived facts and offer timestamps.
- [ ] No keys enter browser responses, retrieved evidence, or logs.

## Task 6: Freeze and evaluate the baseline

Prepare 10 development and 40 held-out cases, relevance labels, and an independent
catalog enumerator. Add the unseen fifth category only after freezing the system.

- [ ] Snapshot/case splits and three repeats per core configuration are reproducible.
- [ ] Retrieval, single-call, agent, and serving comparisons keep their stated controls.
- [ ] Reports follow `docs/evaluation-metrics.md`, including quality, cost, failure counts, timing components, and at least one failure trace; streaming serving measurements are separate from the current blocking-client duration.

**Verification:** Scoring/enumerator self-checks, fixture replay, and a complete
baseline run; manually review evidence labels and preference-fit ratings.
**Dependencies:** 2, 3, 4; 5 for whole-application timings. **Scope:** Medium.
**Likely files:** `evaluations/cases.jsonl`, `evaluations/run.py`,
`evaluations/score.py`, `tests/test_evaluation.py`, `docs/experiments/baseline.md`.

## Checkpoint before model changes

- [ ] Baseline results and snapshot IDs are archived.
- [ ] Held-out cases, derivatives, and category leakage are excluded from training/calibration.
- [ ] Errors identify acquisition, retrieval, arithmetic, tool-use, or serving causes.

## Task 7: Run an SFT pilot

Choose a compatible trainable checkpoint after resource inspection and curate
development-only demonstrations. DPO is a later optional experiment.

- [ ] A small training run saves and reloads an adapter with its base-model revision and tokenizer.
- [ ] The SFT model and its own base model run the same frozen evaluation workload.
- [ ] Tool validity and task quality are reported without claiming cross-model gains as an SFT effect.

**Verification:** Data split checks, train/export/reload smoke test, and a paired
base-versus-SFT evaluation. Record an explicit resource limitation if infeasible.
**Dependencies:** 6. **Scope:** Medium.
**Likely files:** `training/sft.py`, `configs/sft.yaml`,
`training/prepare_data.py`, `tests/test_training_data.py`, `docs/experiments/sft.md`.

## Task 8: Quantize and benchmark serving

Calibrate supported INT8/INT4 variants of the same selected checkpoint and test
one batching, caching, parallelism, or execution-mode change at a time.

- [ ] Each supported artifact reloads and records its exact model/calibration/runtime provenance.
- [ ] Quality comparisons use the same model, snapshot, prompts, and generation settings.
- [ ] Reports include memory, TTFT, throughput, end-to-end latency, and success tradeoffs.

**Verification:** Artifact load/format checks and repeated controlled runs with
failure accounting; report unsupported methods rather than silently changing models.
**Dependencies:** 6; 7 when evaluating an SFT checkpoint. **Scope:** Medium.
**Likely files:** `quantization/quantize.py`, `configs/quantization.yaml`,
`deploy/tacc/qwen-public-serve/job.sh`, `evaluations/serving.py`,
`docs/experiments/optimization.md`.

## Final checkpoint

- [ ] The end-to-end system and its supported runtime are reproducible.
- [ ] Results distinguish goals, completed work, failed combinations, and limitations.
- [ ] The report contains measured charts only after experimental data exist.
