# Improvements Backlog

Work through these in order by priority. Each item is isolated and can be done independently.

---

## ✅ Done

- ~~Concurrent Textract + KB calls~~
- ~~Textract confidence filtering~~
- ~~Token budget guard~~
- ~~Response caching (Postgres `document_cache`)~~
- ~~API key authentication (`X-API-Key` header)~~

---

## 🔴 P0 — Critical: Unhandled Exceptions (Silent 500s)

### ~~1. `db_service` — no exception handling on DB calls~~ ✅ Done

**Files:** `services/db_service.py`, `api/routes.py`

`get_cached_doc`, `save_cached_doc`, and `ensure_table` all call `psycopg.connect()` with
zero try/except. A DB outage produces an unhandled `psycopg.OperationalError` that bypasses
the route-level `RuntimeError` catch and returns a generic, unstructured `500`.

**Fix:**
- Wrap all `psycopg.connect` blocks in try/except, catch `psycopg.Error`, log it, re-raise as `RuntimeError`.
- Add a `try/except RuntimeError` in `_handle_decode` around both `get_cached_doc` and `save_cached_doc`,
  returning `HTTP_503` with `stage: "cache"`.

---

### ~~2. `kb_service` — no exception handling on Bedrock or PGVector calls~~ ✅ Done

**File:** `services/kb_service.py`

`retrieve_relevant_law` creates a fresh boto3 Bedrock client and PGVector connection on every
call with no try/except. A Bedrock auth error (`ClientError`) or PGVector connection failure
(`psycopg.OperationalError`) is not a `RuntimeError`, so it bypasses the route-level catch
entirely and returns an unstructured `500` with no log message.

**Fix:**
- Wrap the Bedrock client creation, embeddings call, and `similarity_search_with_score` in a
  try/except catching `ClientError`, `BotoCoreError`, and `Exception`-narrowed DB errors.
- Re-raise as `RuntimeError` with a descriptive message so the route's existing `503` handler picks it up.

---

### 3. `chat_service` + `draft_service` — LLM `invoke()` is unguarded

**Files:** `services/chat_service.py`, `services/draft_service.py`

`llm.invoke(...)` in both services has no try/except. Network timeouts, Gemini API errors,
or Bedrock `ThrottlingException`s are not `RuntimeError`s, so they bypass the route's
`HTTP_502` catch and surface as unformatted `500` responses with no log entry.

**Fix:**
- Wrap `llm.invoke(...)` in both services in try/except.
- Catch `Exception` (narrowed to LangChain/boto3 error types where possible), log the error,
  re-raise as `RuntimeError('LLM generation failed: ...')`.

---

### ~~4. `kb_service` — silent mock data fallback masks real failures~~ ✅ Done

**File:** `services/kb_service.py` (lines 46–55)

When `similarity_search_with_score` returns zero results (which also happens on a DB
connection failure), the code silently falls back to a **hardcoded Section 73 fabrication**
and returns `('CGST Act 2017',)` as the source. A broken DB connection produces a
"grounded" legal response citing invented mock data.

**Fix:**
- Remove the mock fallback entirely.
- Return `([], [])` when no results are found.
- The existing empty-corpus guards in `chat_service` and `draft_service` already handle this
  correctly with a safe "insufficient information" response.

---

## 🟡 P1 — Medium: Robustness Issues

### 5. `textract_service` — infinite polling loop with no timeout

**File:** `services/textract_service.py` (lines 62–67)

The `while True: time.sleep(1)` Textract job polling has no timeout. If AWS returns a job
that stays `IN_PROGRESS` indefinitely (large document, service-side bug), the request thread
blocks forever, holding a thread pool slot.

**Fix:**
- Add a `MAX_POLL_SECONDS` constant (e.g. `300`).
- Track elapsed time with `time.monotonic()` and raise `RuntimeError('Textract job timed out')` if exceeded.

---

### 6. `draft_service` — bare `except Exception: pass` on JSON parse

**File:** `services/draft_service.py` (line 138)

The JSON parsing block uses `except Exception: pass`, which is explicitly banned by the
code style guide. Unexpected parse failures are silently swallowed with no log message,
and execution falls through to the regex fallback silently.

**Fix:**
- Replace with `except json.JSONDecodeError as exc:`.
- Add `logger.warning('LLM output was not valid JSON for %s: %s', document_id, exc)` before falling through.

---

### 7. Hardcoded model name in `draft_service`

**File:** `services/draft_service.py` (line 88), `config.py`

`'global.amazon.nova-2-lite-v1:0'` is hardcoded directly in the `ChatBedrockConverse`
constructor. This violates the rule against magic strings and makes model changes require
a code edit instead of a config change.

**Fix:**
- Add `bedrock_model_id: str = Field(default='global.amazon.nova-2-lite-v1:0')` to `Settings` in `config.py`.
- Reference `settings.bedrock_model_id` in `draft_service`.

---

### 8. `kb_service` — new boto3 client and PGVector instance on every request

**File:** `services/kb_service.py`

Unlike `textract_service` (which uses a module-level singleton), `kb_service` creates a fresh
`bedrock-runtime` boto3 client and a new `PGVector` connection on every call to
`retrieve_relevant_law`. This adds significant latency and connection overhead per request.

**Fix:**
- Extract a `_get_vector_store()` function that caches the `PGVector` instance and boto3
  client at module level (same pattern as `_get_client()` in `textract_service`).

---

### 9. Inline imports inside function bodies

**Files:** `services/draft_service.py` (lines 81, 127), `services/kb_service.py` (line 20)

`import boto3` and `import json` are inside function bodies. `json` is a stdlib module that
must always be at the top. The boto3 imports hide the service's dependencies and are
re-evaluated conceptually on every call, even if Python caches them.

**Fix:** Move all imports to the module top level, in the correct alphabetical block order.

---

## 🟢 P2 — Low Priority / Hygiene

### 10. Request body size limit

**File:** `main.py`

FastAPI defaults to a 1MB body limit. The `/api/v1/ask` endpoint only receives S3 keys and
short text fields — a 10KB limit would be more than enough and would prevent abuse.

**Fix:** Add `app.router.limit("10KB")` or configure via a Starlette middleware.

---

### 11. `ensure_table` startup failure is swallowed silently

**File:** `main.py` (lines 48–51)

The `except Exception` block in the lifespan handler logs the error but continues startup.
The app starts and accepts requests even though the `document_cache` table may not exist,
causing the first decode request to fail at the DB write stage with a confusing error.

**Fix:** Either re-raise the exception to abort startup, or add a clear warning that decode
mode is degraded and cache writes will fail.

---

## 🧪 Tests Needed

### 12. Unit test — JSON parser in `draft_service`

**File:** `tests/test_draft_service.py`
Test the JSON extraction path against: clean JSON, JSON with newlines, malformed JSON
(should log warning and fall through to regex fallback).

### 13. Unit test — Textract line filtering

**File:** `tests/test_textract_service.py`
Mock the boto3 response and verify WORD blocks are excluded, LINE blocks are joined,
and low-confidence lines are dropped correctly.

### 14. Integration test — error paths return correct HTTP codes

**File:** `tests/test_routes.py`
Use `moto` + `pytest` to verify:
- DB failure during cache lookup → `503` with `stage: "cache"`
- Textract failure → `422` with `stage: "textract"`
- KB failure → `503` with `stage: "knowledge_base"`
- LLM failure → `502` with `stage: "llm_generation"`
