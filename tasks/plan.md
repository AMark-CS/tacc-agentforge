# Shopping and Budget Advisor implementation plan

Updated 2026-10-26 from the supplied `proposal.docx`. Implementation remains
planned. Task acceptance checks are recorded in [todo.md](todo.md).

## Objective and scope

Build a shopping advisor that retrieves source-linked evidence, compares
products or baskets, and validates requirements and available funds. The first
deliverable is a complete request from the browser to the TACC worker and back,
using a reproducible knowledge snapshot. SFT and quantization follow that baseline.

The pilot targets approximately 100 products and 300 review excerpts where
available, 10 development cases, and 40 held-out cases. Eight held-out cases cover
a fifth category added after the agent, prompts, and tools are frozen. These are
planned dataset sizes, not collected data or measured results.

## Architecture decisions

- Add acquisition and knowledge as a distinct capability layer alongside the
  application, post-training, inference, evaluation, and TACC layers.
- Use SerpApi Google Shopping for discovery and Google Product for enrichment.
  Follow returned product identifiers or page tokens, rather than constructing
  undocumented detail URLs. Prices, merchant offers, specifications, and review
  text are optional observations; a review count is not review text.
- Keep the SerpApi key and all public API calls in the application backend.
  Workers query local snapshots. A missing-evidence request can return to the
  backend for budgeted acquisition before the worker resumes. Do not assume
  outbound internet access from TACC compute nodes.
- Reuse the current TACC serving configuration and standard-library model client.
  Verify Windows upload/download paths before extending the request/result bridge.
  The existing successful login does not establish working job submission.
- Use SQLite and FTS5 first. Structured fields handle hard requirements and prices;
  full-text search retrieves relevant specifications and reviews. Add an embedding
  model or separate vector service only if labeled retrieval failures justify it.
- Keep a persistent knowledge base and immutable evaluation snapshots. Temporary
  per-session caches and provider-side archives do not replace either one.
- Keep product facts in the knowledge base. Train model behavior using development
  examples; exclude held-out cases, their derivatives, and the unseen category from
  SFT, preferences, calibration, and prompt tuning.

## Data and evidence contract

| Record | Minimum fields and behavior |
| --- | --- |
| Acquisition | Request ID, engine, query, country/language/location, search ID, acquisition time, provider processing time when returned, response status, sanitized raw-response path and hash |
| Product variant | Internal product/variant IDs, category, brand/model, distinguishing attributes, units, source IDs; never merge variants by title alone |
| Merchant offer | Variant ID, merchant, source URL, exact decimal price, explicit currency, region, quantity/unit, shipping/tax observations or unknown, observation time |
| Evidence | Evidence ID, variant/offer ID, source URL, excerpt or attribute, source/acquisition times, content hash; deduplicate repeated excerpts while retaining conflicts |
| Snapshot | Snapshot ID, schema version, record counts, manifest/hashes, index version, locale, freshness policy; immutable once used for evaluation |
| Recommendation | Request and snapshot IDs, cited evidence IDs, quantities, cost components, verified/unknown fields, available funds, remaining funds, tradeoffs or clarification |

An ingestion timestamp does not prove a price is fresh. Preserve provider cache
metadata when available, display price age, and apply an explicit freshness policy.
Do not infer currency from an ambiguous symbol or silently convert currencies.
Refresh only within a bounded API-call budget; missing essential cost information
prevents an affordability claim. Save source URLs without API keys or signed secret
parameters, and keep personal spending/reserve values out of external search queries.

## Development dependency chain

0. **Transport gate:** Verify SDK file round-trip and worker prerequisites. This
   can precede and run independently of initial SerpApi adapter development.
1. **Collect:** Connect SerpApi, validate optional fields and failures, archive
   sanitized responses, and audit representative records.
2. **Build the KB:** Normalize variants and offers, attach provenance, index
   evidence, and export a replayable snapshot.
3. **Integrate:** Complete budget validation and the bounded tool loop, then
   connect browser/backend requests to the worker and return cited plans.
4. **Evaluate:** Freeze development/test boundaries, label relevance, establish
   task ground truth, and run baseline and failure comparisons.
5. **Post-train:** Select a compatible trainable model, run a small SFT pilot,
   export/reload the adapter, and compare against its own base model. DPO requires
   usable preference pairs and remaining resources.
6. **Optimize:** Quantize the same selected checkpoint and tune one serving factor
   per experiment. Keep the course 27B INT4 model as a deployment reference.

Checkpoints after collection/indexing, after the complete user request, and after
evaluation prevent training and serving work from proceeding without a baseline.
No calendar dates or parallel-agent assignments are assumed.

## Evaluation controls

Full metric definitions, denominators, timing boundaries, baseline controls,
and Ragas/vLLM references are in [the evaluation design](../docs/evaluation-metrics.md).

- Replay 40 held-out cases three times per core configuration, with model settings
  and evidence snapshots fixed. Interleave configurations; separate queue, bridge,
  acquisition, retrieval, model, and whole-task time. Report API counts/errors and
  cached versus refreshed requests separately.
- Compare filtered FTS5, keyword-only, and no additional textual evidence with the
  same structured candidates, prices, model, and budget tools. Relevance labels
  support Recall@5 and MRR@5; source review supports claim accuracy and coverage.
- Compare the bounded agent with a single-call model receiving the same candidate
  evidence. For all tasks, independently enumerate the finite snapshot catalog;
  a no-plan conclusion applies only to that catalog, not to every product online.
- Compare sequential/two-way analysis and filtered/unfiltered context separately.
  Use the same base model for SFT ablations and the same trained checkpoint for
  precision ablations. Record unsupported combinations rather than equating
  differently sized models with a controlled quantization experiment.
- Target zero budget violations and arithmetic errors. Two reviewers score fit
  from 1 to 5; report rating agreement and paired per-case outcome changes.
  The efficiency target is 20% lower median latency or 30% fewer input tokens with
  at most a five-percentage-point success loss. These are goals, not guarantees.
- Inject timeouts, quota errors, malformed outputs, missing costs, stale prices,
  conflicting reviews, and instruction-like text in evidence. Archive at least one
  full failure trace. Do not drop failed requests from latency/success reporting.

## Resource and integration risks

| Risk | Handling and decision gate |
| --- | --- |
| Incomplete SerpApi fields or review coverage | Audit returned fields first; retain missing values and report coverage, rather than fabricating 300 excerpts |
| API quota and acquisition latency | Bounded retries/call budget, query caching, recorded backoff, and frozen evaluation replay; confirm account quota before collection |
| Native Windows SDK paths | Round-trip gate before bridge work; document the tested client environment and isolate any compatibility fix |
| Two concurrent jobs and one-hour allocations | One GPU service plus one CPU worker; persist outputs in USERDIR and resume bounded stages after restart |
| Training feasibility and model identity | Inspect actual model config, original weights, GPU memory, runtime support, and licensing before choosing a trainable checkpoint |
| Price drift or mixed variants | Store offer timestamps and locale; match variants explicitly; clarify unknown essential facts |
| Retrieved text steering tools | Schema-checked tool calls, allowlisted read-only tools, bounded steps, and evidence separated from instructions |

## Two-page proposal design

The exported proposal keeps the original English title, course label, latest
author line, three-section structure (Motivation and task, Proposed system,
Evaluation plan), budget definition, test scale, and resource-dependent
post-training scope. Paragraph lead-ins follow the original proposal.
It replaces the temporary-cache-only description with a persistent, versioned KB.
The original DOCX is preserved.

| Visual | Purpose | Space allocation |
| --- | --- | --- |
| Six-layer architecture | Make data acquisition/KB first-class while retaining application, training, inference, evaluation, and infrastructure | Page 1 within Proposed system |
| Baseline and ablation table | Compare retrieval/agent configurations and reserve TSR, faithfulness, violation, and latency result columns without invented scores | Top of page 2 |
| Pipeline metric table | Assign deterministic checks, Ragas, and vLLM measures to the six evaluation stages | Middle of page 2 |

The six-stage development chain remains a separate SVG; its order is summarized
in page 1 prose. Avoid a duplicate workflow within the PDF page budget. After
measurements exist, use quality-versus-latency or quality-versus-VRAM plots in a
results report. Do not include invented bars, curves, or unlabeled synthetic data.

The canonical source is `docs/proposal/latex/main.tex`. The PDF uses US Letter
(215.9 x 279.4 mm), `newtxtext` and `newtxmath` Times-style fonts, 11 pt body,
no paragraph indent, compact spacing, and margins of 1.65 cm left/right/bottom
and 1.52 cm top. Text/tables are black and white; the architecture remains colored.
The original DOCX is not overwritten. Build with installed pdfLaTeX and bundled Python:

```powershell
& 'C:/Users/Mark/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' docs/proposal/build_proposal.py
```

The builder compiles LaTeX twice, checks exact Letter page dimensions, required
sections, figure/table placement, overfull-box warnings, and a two-page limit.
The PDF embeds vector graphics and fonts. Render both pages with Poppler and
inspect them after edits. The existing standalone SVGs remain README assets;
the PDF architecture is rendered natively with TikZ and the newtx fonts.

## Technical sources

- [SerpApi Google Shopping API](https://serpapi.com/google-shopping-api)
- [SerpApi Google Product API](https://serpapi.com/google-product-api)
- [SerpApi Search Archive API](https://serpapi.com/search-archive-api)

Documentation checked on 2026-10-26. Actual account access, dataset collection,
TACC transfer, and end-to-end execution remain to be tested.
