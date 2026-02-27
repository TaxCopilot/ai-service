# TaxCopilot — AI Microservice

The Python AI backend for TaxCopilot. Analyses Indian income tax notices using AWS Textract (OCR), Amazon Bedrock Knowledge Bases (RAG), and Claude (LLM generation).

## How It Works

```
POST /api/v1/decode-notice
        │
① Textract      →  OCR the notice PDF from S3 (skipped if text is sent directly)
② Knowledge Base →  Retrieve relevant law excerpts via semantic search
③ Claude         →  Summarise the notice and draft a formal CA reply
        │
        ↓
{ generated_summary, draft_response, sources_cited }
```

## Prerequisites

- Python 3.11+
- AWS account with the following IAM permissions:
  - `textract:DetectDocumentText`
  - `s3:GetObject` (on your notices bucket)
  - `bedrock:Retrieve` (on your Knowledge Base)
  - `bedrock:InvokeModel` (on the Claude model)

## Setup

```bash
# 1. Create venv and install all dependencies
make setup

# 2. Configure environment variables
cp .env.example .env
# Edit .env — fill in your AWS credentials and BEDROCK_KNOWLEDGE_BASE_ID
```

> **No `make`?** Run the steps manually:
>
> ```bash
> python -m venv venv && venv\Scripts\activate   # Windows
> pip install -r requirements.txt
> ```

## AWS Knowledge Base Setup (one-time)

1. Upload your tax law PDFs to an S3 bucket (e.g. `s3://taxcopilot-knowledge/`)
2. Go to **AWS Console → Amazon Bedrock → Knowledge Bases → Create**
3. Point it at the S3 bucket and click **Sync**
4. Copy the Knowledge Base ID into `.env` as `BEDROCK_KNOWLEDGE_BASE_ID`

## Running Locally

```bash
uvicorn main:app --reload --port 8001
```

The service will be available at `http://localhost:8001`.
Interactive API docs: `http://localhost:8001/docs`

To trigger the AI pipeline locally:

```bash
curl -X POST http://localhost:8001/api/v1/decode-notice \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-super-secret-key-123" \
  -d '{
    "document_id": "test_123",
    "notice_type": "143(1)",
    "s3_bucket": "your-bucket-name",
    "s3_key": "tax_notice_sample.pdf"
  }'
```

_(Note: If `API_KEY` is not set in `.env`, the `X-API-Key` header is optional during local development)._

## API

### `POST /api/v1/decode-notice`

**With a PDF in S3 (Textract will OCR it):**

```json
{
  "document_id": "doc_abc123",
  "notice_type": "143(1)",
  "s3_bucket": "taxcopilot-notices",
  "s3_key": "uploads/notice.pdf"
}
```

**With pre-extracted text (Textract is skipped):**

```json
{
  "document_id": "doc_abc123",
  "notice_type": "143(1)",
  "extracted_text": "Dear Assessee, you have been assessed..."
}
```

**Response:**

```json
{
  "document_id": "doc_abc123",
  "generated_summary": "This notice under Section 143(1)...",
  "draft_response": "Dear Assessing Officer...",
  "sources_cited": ["s3://taxcopilot-knowledge/income-tax-act.pdf"]
}
```

### `GET /health`

Returns `200 OK` when the service is running.

## Project Structure

```
ai-services/
├── config.py                  # Centralised settings via Pydantic
├── main.py                    # FastAPI app entry point
├── api/
│   └── routes.py              # POST /api/v1/decode-notice
└── services/
    ├── textract_service.py    # AWS Textract OCR
    ├── kb_service.py          # Bedrock Knowledge Base retrieval
    └── draft_service.py       # Claude generation + JSON parsing
```

## Tech Stack

| Component     | Service                                        |
| ------------- | ---------------------------------------------- |
| OCR           | AWS Textract                                   |
| Vector search | Amazon Bedrock Knowledge Bases                 |
| LLM           | Amazon Bedrock — `anthropic.claude-sonnet-4-6` |
| Framework     | FastAPI + Uvicorn                              |
| Validation    | Pydantic v2                                    |
