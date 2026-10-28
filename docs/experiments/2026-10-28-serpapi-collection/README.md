# SerpApi collection and coverage audit — 2026-10-28

First live collection for the shopping advisor (todo Task 1). Run on a TACC CPU node (job 27837), with the
audit run as job 27840. The raw responses (5.3 MB) stay private on TACC in
`/mnt/home/markzhang/USERDIR/data/serpapi/run-27837/`; only the summary and the audit are committed here.

## What was collected

- 8 Google Shopping searches (4 categories x 2 queries: earbuds, chargers, keyboards, backpacks), US locale
  (`gl=us`, `hl=en`), 40 results each.
- 40 Google Product detail lookups (the top 5 results of each search, deduplicated).
- 48 API calls of the configured limit of 50. Every call returned HTTP 200.

## Coverage (from `audit-27837.json`)

| Item | Result |
| --- | --- |
| Shopping rows | 320 rows, 308 unique product IDs |
| Rows with price / product ID / source | 100% / 100% / 100% |
| Rows with rating and review count | 85% (271 of 320) |
| Rows with delivery text | 61% |
| Price range | $1.64 to $4,006, median $49.99; 319 rows in `$`, one in `£` |
| Detail lookups that returned data | 27 of 40 |
| Detail pages with merchant offers | 27 of 27 (3-5 offers each, 86 in total) |
| Offers with price, total and extra costs | 100%; shipping text 91%; numeric shipping 36%; rating 58% |
| Detail pages with attributes / description / price insights | 27 / 24 / 27 of 27 (median 6 attributes) |
| **Detail pages with review text** | **0 of 27**: only review counts appear |

## Findings

1. **13 of 40 lookups returned "Google Product hasn't returned any results for this query".** The client passed the
   bare `product_id`. Every shopping row also carries an `immersive_product_page_token` and a ready-made product
   API link built on it, and all 13 failed products have that token. The client now prefers the token
   (`get_product(page_token=...)`). The 13 lookups have not been retried, so it is not yet shown that the token
   fixes them.
2. **No review text was returned.** The pilot target in `tasks/plan.md` is about 300 review excerpts "where
   available". With these calls the data has review counts and ratings, not excerpts. The plan already says a
   review count is not review text; reaching review text probably needs a different call and is untested.
3. **Currency is mixed in at least one row** (`£`), so prices must carry an explicit currency and not be
   compared blindly.
4. **Offers are rich enough for budget tasks:** price, total, additional costs and shipping per merchant, which
   supports exact budget arithmetic where those fields are present.

## Safety

The client enforces a hard call budget (retries included) and redacts the API key from every saved response and the
call index. The key lives in `secrets/serpapi_key` on TACC with owner-only permissions and was not read by the
assistant. 14 unit tests pass on TACC (job 27841).

## Not done

- Retry the 13 failed lookups with page tokens.
- Find a way to get review text.
- Check the credits used on the SerpApi dashboard (48 requests were sent; whether empty lookups were charged is unknown).
- Normalize into variants, offers and evidence (todo Task 2).

## Files

`summary-27837.json`, `audit-27837.json`; code in `knowledge/serpapi.py`, `deploy/tacc/serpapi-collect/`,
`configs/shopping_queries.json`, `tests/test_serpapi.py`.
