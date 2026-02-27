# Improvements Backlog

Work through these in order — each item is isolated and can be done independently.

---

## Optimizations

### ~~1. Concurrent Textract + KB calls~~ ✅ Done

### ~~2. Textract confidence filtering~~ ✅ Done

### ~~3. Token budget guard~~ ✅ Done

### ~~4. Response caching~~ ✅ Done

---

## Safety Measures

### ~~5. API key authentication~~ ✅ Done

### 6. Request size limit

**File:** `main.py`
Lower FastAPI's default 1MB body limit — the request only carries S3 keys, so 10KB is more than enough. Prevents abuse.

### 7. IAM least-privilege reminder

**File:** `README.md` (document only)
Ensure the attached IAM policy grants only:

- `textract:DetectDocumentText`
- `s3:GetObject` scoped to the notices bucket
- `bedrock:Retrieve` scoped to the KB ARN
- `bedrock:InvokeModel` scoped to the Claude model ARN

### 8. Secrets never in code

**Verify:** All files
Confirm no AWS credentials, KB IDs, or model IDs are hardcoded anywhere — all must come from `settings`.

---

## Tests

### 9. Unit test — JSON parser

**File:** `tests/test_draft_service.py`
Test `_parse_json` from `draft_service.py` against:

- Clean JSON
- JSON wrapped in markdown fences
- Malformed JSON (should raise `ValueError`)

### 10. Unit test — Textract line filtering

**File:** `tests/test_textract_service.py`
Mock the boto3 response and verify that WORD blocks are excluded and LINE blocks are joined correctly.

### 11. Integration test — full pipeline (mocked AWS)

**File:** `tests/test_routes.py`
Use `moto` to mock Textract and patch the KB + Bedrock calls.
Send a `NoticeRequest` to the FastAPI test client and assert:

- Status 200
- `NoticeResponse` fields are non-empty
- `document_id` matches the request

### 12. Integration test — error paths

**File:** `tests/test_routes.py`
Verify the correct HTTP status codes are returned:

- Textract failure → 422
- KB failure → 503
- Bedrock failure → 502
