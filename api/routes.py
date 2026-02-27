"""
POST /api/v1/decode-notice — full pipeline: Textract → Knowledge Base → Bedrock.
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.draft_service import NoticeResponse, generate_notice_reply
from services.kb_service import retrieve_relevant_law
from services.textract_service import extract_text_from_s3

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Notice Decoder"])


class NoticeRequest(BaseModel):
    """
    Sent by the TypeScript backend after uploading the notice PDF to S3.
    This service handles OCR internally via Textract.
    """

    document_id: str = Field(examples=["doc_abc123"])
    notice_type: str = Field(description="e.g. '143(1)', 'ASMT-10'", examples=["143(1)"])
    s3_bucket: str = Field(examples=["taxcopilot-notices"])
    s3_key: str = Field(examples=["uploads/notice_abc123.pdf"])


@router.post("/decode-notice", response_model=NoticeResponse, summary="Decode a tax notice")
async def decode_notice(request: NoticeRequest) -> NoticeResponse:
    logger.info(
        "decode-notice | id=%s type=%s src=s3://%s/%s",
        request.document_id, request.notice_type, request.s3_bucket, request.s3_key,
    )

    # Step 1 — OCR
    try:
        extracted_text = extract_text_from_s3(request.s3_bucket, request.s3_key)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"stage": "textract", "error": str(exc)}) from exc

    # Step 2 — Semantic law retrieval
    try:
        passages, sources = retrieve_relevant_law(extracted_text)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail={"stage": "knowledge_base", "error": str(exc)}) from exc

    retrieved_law = "\n\n---\n\n".join(passages)
    unique_sources = list(dict.fromkeys(sources))  # deduplicate, preserve order

    # Step 3 — LLM generation
    try:
        result = generate_notice_reply(
            document_id=request.document_id,
            extracted_text=extracted_text,
            retrieved_law=retrieved_law,
            sources_cited=unique_sources,
        )
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail={"stage": "bedrock_generation", "error": str(exc)}) from exc

    logger.info("decode-notice complete | id=%s", request.document_id)
    return result
