"""
POST /api/v1/decode-notice — full pipeline: Textract → Knowledge Base → Bedrock.

All AWS calls are blocking (boto3 is sync-only). We wrap each in
run_in_executor so the FastAPI event loop stays free to handle other
requests while waiting on AWS network I/O.
"""

import asyncio
import logging
from collections.abc import Callable
from functools import partial
from typing import TypeVar

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from services.draft_service import NoticeResponse, generate_notice_reply
from services.kb_service import retrieve_relevant_law
from services.textract_service import extract_text_from_s3

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/v1', tags=['Notice Decoder'])

_PASSAGE_SEPARATOR = '\n\n---\n\n'
_CACHE_MAX_ENTRIES = 100

_response_cache: dict[str, NoticeResponse] = {}

R = TypeVar('R')


class NoticeRequest(BaseModel):
    """
    Two input modes are supported:

    1. PDF in S3 — provide s3_bucket + s3_key. Textract will OCR the file.
    2. Pre-extracted text — provide extracted_text directly. Textract is skipped.

    At least one mode must be supplied; both can be provided (S3 takes priority).
    """

    document_id: str = Field(examples=['doc_abc123'])
    notice_type: str = Field(description="e.g. '143(1)', 'ASMT-10'", examples=['143(1)'])

    # PDF path in S3 — triggers Textract OCR when present
    s3_bucket: str | None = Field(default=None, examples=['taxcopilot-notices'])
    s3_key: str | None = Field(default=None, examples=['uploads/notice_abc123.pdf'])

    # Pre-extracted text — skips Textract entirely
    extracted_text: str | None = Field(default=None, examples=['Dear Assessee, ...'])

    @model_validator(mode='after')
    def check_at_least_one_input(self) -> 'NoticeRequest':
        has_s3 = bool(self.s3_bucket and self.s3_key)
        has_text = bool(self.extracted_text and self.extracted_text.strip())
        if not has_s3 and not has_text:
            raise ValueError(
                "Provide either (s3_bucket + s3_key) to trigger OCR, "
                'or extracted_text to skip it.'
            )
        return self


async def _run(fn: Callable[..., R], *args: object) -> R:
    """Run a blocking function in the default thread pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


@router.post('/decode-notice', response_model=NoticeResponse, summary='Decode a tax notice')
async def decode_notice(request: NoticeRequest) -> NoticeResponse:
    cached = _response_cache.get(request.document_id)
    if cached is not None:
        logger.info('Cache hit | id=%s', request.document_id)
        return cached

    logger.info(
        'decode-notice | id=%s type=%s has_s3=%s has_text=%s',
        request.document_id,
        request.notice_type,
        bool(request.s3_bucket),
        bool(request.extracted_text),
    )

    # Step 1 — OCR (only when a file is provided)
    if request.s3_bucket and request.s3_key:
        try:
            extracted_text: str = await _run(
                extract_text_from_s3, request.s3_bucket, request.s3_key
            )
        except RuntimeError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={'stage': 'textract', 'error': str(exc)},
            ) from exc
    else:
        extracted_text = request.extracted_text  # type: ignore[assignment]
        logger.info('Textract skipped — using pre-extracted text | id=%s', request.document_id)

    # Step 2 — Semantic law retrieval (non-blocking)
    try:
        passages, sources = await _run(retrieve_relevant_law, extracted_text)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'knowledge_base', 'error': str(exc)},
        ) from exc

    retrieved_law = _PASSAGE_SEPARATOR.join(passages)
    unique_sources = list(dict.fromkeys(sources))  # deduplicate, preserve order

    # Step 3 — LLM generation (non-blocking)
    try:
        result: NoticeResponse = await _run(
            generate_notice_reply,
            request.document_id,
            extracted_text,
            retrieved_law,
            unique_sources,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={'stage': 'bedrock_generation', 'error': str(exc)},
        ) from exc

    logger.info('decode-notice complete | id=%s', request.document_id)

    # Evict oldest entry if cache is full, then store the new result.
    if len(_response_cache) >= _CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_response_cache))
        del _response_cache[oldest_key]
    _response_cache[request.document_id] = result

    return result
