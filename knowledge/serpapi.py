"""SerpApi acquisition (todo Task 1): Google Shopping discovery and Google Product enrichment.

Standard library only. Design rules, from tasks/plan.md:

- The API key is read from a file and never logged, printed or saved; saved responses and the call
  index are redacted before they are written.
- Every attempt, including retries, counts toward a hard call budget (each attempt may cost a
  search credit), so a bug cannot burn through the quota.
- Each call is archived with its query, locale, search ID, timestamps and a content hash. Missing
  fields, errors and quota failures are recorded explicitly, never filled in.
- Product detail calls follow the product IDs returned by a shopping search; no detail URLs are built.
"""
import hashlib
import json
import os
import re
import ssl
import stat
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://serpapi.com/search.json"
CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"  # the nodes' Python does not trust the certs without it


class BudgetExceeded(Exception):
    """The configured number of API calls has been used."""


class QuotaExhausted(Exception):
    """SerpApi reports the account has no searches left (or is rate limiting persistently)."""


class AuthenticationFailed(Exception):
    """SerpApi rejected the API key."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_key(path: Path) -> str:
    """Read the key file; tighten its permissions to owner-only if they are looser."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"API key file not found: {path}")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        os.chmod(path, 0o600)
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def redact(text: str, key: str) -> str:
    """Remove the key value and any api_key=... query parameter from text."""
    if key:
        text = text.replace(key, "REDACTED")
    return re.sub(r"(api_key=)[^&\"\s]+", r"\1REDACTED", text)


@dataclass
class CallRecord:
    """One archived API call (an 'Acquisition' record in tasks/plan.md)."""
    call_id: str
    engine: str
    params: dict                    # request parameters, without the API key
    requested_at: str
    status: str                     # "ok" or "error"
    http_status: int | None = None
    error: str = ""
    search_id: str = ""             # search_metadata.id returned by SerpApi
    processed_at: str = ""          # provider processing time, when returned
    api_seconds: float | None = None
    raw_path: str = ""
    sha256: str = ""
    attempt: int = 1


class SerpApiClient:
    def __init__(self, key: str, out_dir: Path, max_calls: int, locale: dict | None = None,
                 base_url: str = BASE_URL, retries: int = 2, backoff_s: float = 5.0,
                 timeout_s: float = 60.0, sleep=time.sleep):
        self.key, self.out_dir, self.max_calls = key, Path(out_dir), max_calls
        self.locale = dict(locale or {"gl": "us", "hl": "en"})
        self.base_url, self.retries, self.backoff_s = base_url, retries, backoff_s
        self.timeout_s, self.sleep = timeout_s, sleep
        self.calls_made = 0
        self.records: list[CallRecord] = []
        (self.out_dir / "raw").mkdir(parents=True, exist_ok=True)
        cafile = CA_BUNDLE if Path(CA_BUNDLE).exists() else None
        self._ssl = ssl.create_default_context(cafile=cafile)

    # -- low level ---------------------------------------------------------------------------
    def _http_get(self, params: dict) -> tuple[int, str]:
        url = self.base_url + "?" + urlencode({**params, "api_key": self.key})
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_s, context=self._ssl) as response:
                return response.status, response.read().decode("utf-8")
        except HTTPError as exc:
            try:
                return exc.code, exc.read().decode("utf-8", errors="replace")
            finally:
                exc.close()

    def _request(self, engine: str, params: dict) -> tuple[CallRecord, dict | None]:
        params = {**self.locale, **params, "engine": engine}
        record = None
        for attempt in range(1, self.retries + 2):
            if self.calls_made >= self.max_calls:
                raise BudgetExceeded(f"call budget of {self.max_calls} reached")
            self.calls_made += 1  # every attempt counts, retries included
            record = CallRecord(call_id=uuid.uuid4().hex[:12], engine=engine, params=params,
                                requested_at=now_iso(), status="error", attempt=attempt)
            started = time.perf_counter()
            try:
                status, body = self._http_get(params)
            except (URLError, TimeoutError, OSError) as exc:
                record.error = redact(f"network error: {exc!r}", self.key)[:300]
                self._log(record)
                if attempt <= self.retries:
                    self.sleep(self.backoff_s * attempt)
                    continue
                return record, None
            record.http_status = status
            if status in (429,) or status >= 500:
                record.error = f"HTTP {status}"
                self._log(record)
                if attempt <= self.retries:
                    self.sleep(self.backoff_s * attempt)
                    continue
                if status == 429:
                    raise QuotaExhausted("rate limited (HTTP 429) after retries")
                return record, None
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                record.error = "response was not JSON"
                self._log(record)
                return record, None
            self._archive(record, data, time.perf_counter() - started)
            if status in (401, 403):
                raise AuthenticationFailed(f"HTTP {status}: {redact(str(data.get('error', '')), self.key)[:200]}")
            error = str(data.get("error", ""))
            if error:
                record.error = redact(error, self.key)[:300]
                self._log(record)
                if "run out of searches" in error.lower():
                    raise QuotaExhausted(record.error)
                return record, data  # e.g. "Google hasn't returned any results for this query."
            record.status = "ok"
            self._log(record)
            return record, data
        return record, None

    def _archive(self, record: CallRecord, data: dict, seconds: float) -> None:
        meta = data.get("search_metadata") or {}
        record.search_id = str(meta.get("id", ""))
        record.processed_at = str(meta.get("processed_at", ""))
        record.api_seconds = meta.get("total_time_taken", round(seconds, 3))
        text = redact(json.dumps(data, ensure_ascii=False, indent=1), self.key)
        raw = self.out_dir / "raw" / f"{record.call_id}.json"
        raw.write_text(text, encoding="utf-8")
        record.raw_path = str(raw.relative_to(self.out_dir))
        record.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _log(self, record: CallRecord) -> None:
        self.records.append(record)
        line = redact(json.dumps(asdict(record), ensure_ascii=False), self.key)
        with open(self.out_dir / "index.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # -- engines -----------------------------------------------------------------------------
    def search_shopping(self, query: str, **extra) -> tuple[CallRecord, dict | None]:
        return self._request("google_shopping", {"q": query, **extra})

    def get_product(self, product_id: str | None = None, page_token: str | None = None,
                    **extra) -> tuple[CallRecord, dict | None]:
        """Detail lookup. Prefer the page token a shopping result returns: in the first run, 13 of
        40 lookups by bare product_id came back empty even though every result had a token."""
        if not (page_token or product_id):
            raise ValueError("need a page_token or a product_id")
        flags = {"offers": 1, "specs": 1, "reviews": 1}
        ident = {"page_token": page_token} if page_token else {"product_id": product_id}
        return self._request("google_product", {**ident, **flags, **extra})


def top_products(data: dict | None, n: int) -> list[dict]:
    """The first n shopping results that carry a product_id (needed for the detail call)."""
    found = []
    for item in (data or {}).get("shopping_results", []) or []:
        if item.get("product_id"):
            found.append({"product_id": str(item["product_id"]), "title": item.get("title"),
                          "price": item.get("price"), "source": item.get("source"),
                          "page_token": item.get("immersive_product_page_token")})
        if len(found) >= n:
            break
    return found


def collect(client: SerpApiClient, queries: list[dict], products_per_query: int = 5) -> dict:
    """Run every shopping query, then fetch details for the top products (deduplicated).

    Stops cleanly when the budget is used or the quota is exhausted; the summary says why.
    """
    summary = {"queries": 0, "products_requested": 0, "stopped": "", "empty_queries": []}
    seen: set[str] = set()
    try:
        for entry in queries:
            record, data = client.search_shopping(entry["q"], **entry.get("params", {}))
            summary["queries"] += 1
            products = top_products(data, products_per_query)
            if not products:
                summary["empty_queries"].append(entry["q"])
            for product in products:
                if product["product_id"] in seen:
                    continue
                seen.add(product["product_id"])
                client.get_product(product["product_id"], page_token=product.get("page_token"))
                summary["products_requested"] += 1
        summary["stopped"] = "finished"
    except BudgetExceeded as exc:
        summary["stopped"] = f"budget: {exc}"
    except QuotaExhausted as exc:
        summary["stopped"] = f"quota: {exc}"
    except AuthenticationFailed as exc:
        summary["stopped"] = f"authentication: {exc}"
    summary["calls_made"] = client.calls_made
    summary["errors"] = sum(r.status != "ok" for r in client.records)
    return summary
