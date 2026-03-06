'''
POST /api/v1/ask — unified mode-based endpoint.

Two modes:
  chat   (default) — freeform Q&A grounded in the tax law vector store.
  decode            — Textract OCR → RAG → Gemini formal draft reply,
                      with Postgres-backed persistent caching.

All boto3 calls are blocking; we wrap them in run_in_executor so the
FastAPI event loop stays free.
'''

import asyncio
import logging
from collections.abc import Callable
from functools import partial
from typing import Literal, TypeVar, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from api.deps import require_api_key
from services.chat_service import ChatResponse, generate_chat_reply
from services.db_service import get_cached_doc, save_cached_doc
from services.draft_service import NoticeResponse, generate_notice_reply
from services.kb_service import retrieve_relevant_law
from services.textract_service import extract_text_from_s3

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/v1', tags=['TaxCopilot'])

_PASSAGE_SEPARATOR = '\n\n---\n\n'

R = TypeVar('R')


async def _run(fn: Callable[..., R], *args: object) -> R:
    '''Run a blocking function in the default thread pool executor.'''
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    '''
    Unified request for both chat and decode modes.

    chat mode  — set mode="chat" and provide a message.
    decode mode — set mode="decode" and provide s3_bucket + s3_key (or
                  pre-supplied extracted_text) plus a unique document_id.
    '''

    mode: Literal['chat', 'decode'] = Field(
        default='chat',
        description='"chat" for general Q&A, "decode" to analyse a tax notice PDF.',
    )

    # --- chat fields ---
    message: str | None = Field(
        default=None,
        description='User question (required for chat mode).',
        examples=['What is Section 73 of CGST Act?'],
    )

    # --- decode fields ---
    document_id: str | None = Field(
        default=None,
        description='Unique ID of the tax notice (required for decode mode).',
        examples=['doc_abc123'],
    )
    notice_type: str | None = Field(
        default=None,
        description="Notice type, e.g. '143(1)', 'ASMT-10'.",
        examples=['ASMT-10'],
    )
    s3_bucket: str | None = Field(default=None, examples=['taxcopilot-files'])
    s3_key: str | None = Field(default=None, examples=['uploads/notice.pdf'])
    extracted_text: str | None = Field(
        default=None,
        description='Pre-extracted notice text. Skips Textract when provided.',
    )
    regenerate: bool = Field(
        default=False,
        description=(
            'Force a new Gemini draft even if a cached draft exists. '
            'Textract is still skipped if extracted_text is already cached.'
        ),
    )

    @model_validator(mode='after')
    def check_required_fields(self) -> 'AskRequest':
        if self.mode == 'chat':
            if not self.message or not self.message.strip():
                raise ValueError('message is required for chat mode.')
        elif self.mode == 'decode':
            if not self.document_id:
                raise ValueError('document_id is required for decode mode.')
            has_s3 = bool(self.s3_bucket and self.s3_key)
            has_text = bool(self.extracted_text and self.extracted_text.strip())
            if not has_s3 and not has_text:
                raise ValueError(
                    'decode mode requires either (s3_bucket + s3_key) '
                    'or extracted_text.'
                )
        return self


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    '/ask',
    response_model=Union[ChatResponse, NoticeResponse],
    summary='Ask TaxCopilot a question or decode a tax notice',
    dependencies=[Depends(require_api_key)],
)
async def ask(request: AskRequest) -> Union[ChatResponse, NoticeResponse]:  # noqa: UP007

    if request.mode == 'chat':
        return await _handle_chat(request)

    return await _handle_decode(request)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

async def _handle_chat(request: AskRequest) -> ChatResponse:
    '''Retrieve relevant law and generate a freeform chat answer.'''
    message = request.message  # already validated non-empty
    assert message is not None  # noqa: S101 — satisfies type checker

    logger.info('ask | mode=chat | q="%s..."', message[:60])

    try:
        passages, sources = await _run(retrieve_relevant_law, message)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'knowledge_base', 'error': str(exc)},
        ) from exc

    retrieved_law = _PASSAGE_SEPARATOR.join(passages)
    unique_sources = list(dict.fromkeys(sources))

    try:
        result: ChatResponse = await _run(
            generate_chat_reply, message, retrieved_law, unique_sources
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={'stage': 'llm_generation', 'error': str(exc)},
        ) from exc

    logger.info('ask | mode=chat | complete')
    return result


async def _handle_decode(request: AskRequest) -> NoticeResponse:
    '''
    Full decode pipeline with persistent Postgres caching.

    Cache logic:
      - Hit + regenerate=False  → return cached draft immediately ($0 cost).
      - Hit + regenerate=True   → skip Textract, re-run LLM, update cache.
      - Miss                    → full pipeline, save to cache.
    '''
    document_id = request.document_id
    assert document_id is not None  # noqa: S101

    logger.info(
        'ask | mode=decode | id=%s regenerate=%s has_s3=%s has_text=%s',
        document_id,
        request.regenerate,
        bool(request.s3_bucket),
        bool(request.extracted_text),
    )

    # ---- 1. Check persistent cache ----
    try:
        cached = await _run(get_cached_doc, document_id)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'cache', 'error': str(exc)},
        ) from exc

    if cached is not None and not request.regenerate:
        logger.info('ask | mode=decode | cache HIT (full) | id=%s', document_id)
        return NoticeResponse(
            draft_reply=cached['draft_reply'],
            citations=cached['citations'],
            is_grounded=cached['is_grounded'],
        )

    # ---- 2. OCR — skip if we already have the text in cache or request ----
    if cached is not None:
        # regenerate=True but text is already cached — skip Textract
        extracted_text: str = cached['extracted_text']
        logger.info('ask | mode=decode | cache HIT (text only, regenerate) | id=%s', document_id)
    elif request.s3_bucket and request.s3_key:
        try:
            extracted_text = await _run(
                extract_text_from_s3, request.s3_bucket, request.s3_key
            )
        except RuntimeError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={'stage': 'textract', 'error': str(exc)},
            ) from exc
    else:
        extracted_text = request.extracted_text  # type: ignore[assignment]
        logger.info('ask | mode=decode | using pre-supplied extracted_text | id=%s', document_id)

    # ---- 3. RAG retrieval ----
    try:
        passages, sources = await _run(retrieve_relevant_law, extracted_text)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'knowledge_base', 'error': str(exc)},
        ) from exc

    retrieved_law = _PASSAGE_SEPARATOR.join(passages)
    unique_sources = list(dict.fromkeys(sources))

    # ---- 4. LLM draft generation ----
    try:
        result: NoticeResponse = await _run(
            generate_notice_reply,
            document_id,
            extracted_text,
            retrieved_law,
            unique_sources,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={'stage': 'llm_generation', 'error': str(exc)},
        ) from exc

    # ---- 5. Persist to cache ----
    try:
        await _run(
            save_cached_doc,
            document_id,
            extracted_text,
            result.draft_reply,
            result.citations,
            result.is_grounded,
            request.s3_key,
        )
    except RuntimeError as exc:
        # Cache write failure is non-fatal — the response is already generated.
        # Log the error and return the result anyway; next request will regenerate.
        logger.error(
            'ask | mode=decode | cache save FAILED for id=%s: %s',
            document_id,
            exc,
        )

    logger.info('ask | mode=decode | complete | id=%s', document_id)
    return result
