"""
POST /api/v1/ask — unified mode-based endpoint.

Modes:
  chat     (default) — freeform Q&A grounded in the tax law vector store.
  decode             — Textract OCR → RAG → Bedrock formal draft reply,
                       with Postgres-backed persistent caching.
  analyze            — Deep structured analysis of a tax notice (cached per document).
  strategy           — Tailored defence strategy, grounded in RAG + chat history.
  draft              — Full HTML-formatted formal reply draft, built from session context.

All blocking calls are wrapped in run_in_executor so the FastAPI event loop stays free.
Chat history is stored server-side in chat_messages table; clients never send history.
"""

import asyncio
import logging
from collections.abc import Callable
from functools import partial
from typing import Literal, TypeVar, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from api.deps import require_api_key
from services.analysis_service import AnalysisResponse, generate_analysis
from services.chat_service import ChatResponse, generate_chat_reply
from services.db_service import (
    append_message,
    get_analysis_cache,
    get_cached_doc,
    get_chat_history,
    save_analysis_cache,
    save_cached_doc,
)
from services.draft_service import (
    DraftHtmlResponse,
    NoticeResponse,
    generate_html_draft,
    generate_notice_reply,
)
from services.kb_service import retrieve_relevant_law
from services.strategy_service import StrategyResponse, generate_strategy
from services.textract_service import extract_text_from_s3

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/v1', tags=['TaxCopilot'])

_PASSAGE_SEPARATOR = '\n\n---\n\n'

R = TypeVar('R')


async def _run(fn: Callable[..., R], *args: object) -> R:
    """Run a blocking function in the default thread pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """
    Unified request for all AI modes.

    chat     — set mode="chat" and provide a message.
    decode   — set mode="decode" and provide s3_bucket + s3_key (or extracted_text)
               plus a unique document_id.
    analyze  — set mode="analyze" and provide document_id (+ s3 or extracted_text if
               not already in cache from a prior decode run).
    strategy — set mode="strategy" and provide document_id. Optionally provide
               account_details for a personalised strategy.
    draft    — set mode="draft" and provide document_id (s3/text pulled from cache).
    """

    mode: Literal['chat', 'decode', 'analyze', 'strategy', 'draft'] = Field(
        default='chat',
        description='Operation mode.',
    )

    # --- chat fields ---
    message: str | None = Field(
        default=None,
        description='User question (required for chat mode).',
        examples=['What is Section 73 of CGST Act?'],
    )

    # --- shared document fields ---
    document_id: str | None = Field(
        default=None,
        description='Unique ID of the tax notice. Required for decode, analyze, strategy, draft.',
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
        description='Force a fresh LLM call even if a cached result exists.',
    )

    # --- strategy-specific ---
    account_details: str | None = Field(
        default=None,
        description=(
            'Optional taxpayer account details (GSTIN, transaction history, ITC records). '
            'Personalises the strategy when provided.'
        ),
    )

    @model_validator(mode='after')
    def check_required_fields(self) -> 'AskRequest':
        return _validate_for_mode(self)


def _validate_for_mode(req: 'AskRequest') -> 'AskRequest':
    """Validate mode-specific required fields. Separated to keep model clean."""
    if req.mode == 'chat':
        if not req.message or not req.message.strip():
            raise ValueError('message is required for chat mode.')

    elif req.mode == 'decode':
        if not req.document_id:
            raise ValueError('document_id is required for decode mode.')
        _require_text_source(req)

    elif req.mode == 'analyze':
        if not req.document_id:
            raise ValueError('document_id is required for analyze mode.')
        # s3 / text is optional — may already be in cache from a prior decode run

    elif req.mode in ('strategy', 'draft'):
        if not req.document_id:
            raise ValueError(f'document_id is required for {req.mode} mode.')
        # extracted_text must be in cache from a prior analyze/decode run

    return req


def _require_text_source(req: 'AskRequest') -> None:
    """Raises ValueError if neither s3 coords nor extracted_text are provided."""
    has_s3 = bool(req.s3_bucket and req.s3_key)
    has_text = bool(req.extracted_text and req.extracted_text.strip())
    if not has_s3 and not has_text:
        raise ValueError(
            'decode/analyze mode requires either (s3_bucket + s3_key) or extracted_text.'
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    '/ask',
    response_model=Union[
        ChatResponse,
        NoticeResponse,
        AnalysisResponse,
        StrategyResponse,
        DraftHtmlResponse,
    ],
    summary='Ask TaxCopilot a question or process a tax notice',
    dependencies=[Depends(require_api_key)],
)
async def ask(
    request: AskRequest,
) -> Union[
    ChatResponse, NoticeResponse, AnalysisResponse, StrategyResponse, DraftHtmlResponse
]:

    if request.mode == 'chat':
        return await _handle_chat(request)
    if request.mode == 'decode':
        return await _handle_decode(request)
    if request.mode == 'analyze':
        return await _handle_analyze(request)
    if request.mode == 'strategy':
        return await _handle_strategy(request)
    # draft
    return await _handle_draft_html(request)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


async def _handle_chat(request: AskRequest) -> ChatResponse:
    """Retrieve relevant law and generate a freeform chat answer."""
    message = request.message
    assert message is not None  # noqa: S101
    document_id = request.document_id

    logger.info("ask | mode=chat | q='%s...'", message[:60])

    extracted_text = None
    chat_history = []
    
    if document_id:
        try:
            cached_doc = await _run(get_cached_doc, document_id)
            if cached_doc and cached_doc.get('extracted_text'):
                extracted_text = cached_doc['extracted_text']
            chat_history = await _run(get_chat_history, document_id)
        except RuntimeError as exc:
            logger.warning('ask | mode=chat | doc context fetch failed: %s', exc)

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
            generate_chat_reply,
            message,
            retrieved_law,
            unique_sources,
            extracted_text,
            chat_history,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={'stage': 'llm_generation', 'error': str(exc)},
        ) from exc

    if document_id:
        _append_history_safe(document_id, 'user', 'chat', message)
        _append_history_safe(document_id, 'assistant', 'chat', result.answer)

    logger.info('ask | mode=chat | complete')
    return result


async def _handle_decode(request: AskRequest) -> NoticeResponse:
    """
    Full decode pipeline with persistent Postgres caching.

    Cache logic:
      - Hit + regenerate=False  → return cached draft immediately.
      - Hit + regenerate=True   → skip Textract, re-run LLM, update cache.
      - Miss                    → full pipeline, save to cache.
    """
    document_id = request.document_id
    assert document_id is not None  # noqa: S101

    logger.info(
        'ask | mode=decode | id=%s regenerate=%s has_s3=%s has_text=%s',
        document_id,
        request.regenerate,
        bool(request.s3_bucket),
        bool(request.extracted_text),
    )

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

    extracted_text = await _resolve_extracted_text(request, cached, document_id)

    try:
        passages, sources = await _run(retrieve_relevant_law, extracted_text)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'knowledge_base', 'error': str(exc)},
        ) from exc

    retrieved_law = _PASSAGE_SEPARATOR.join(passages)
    unique_sources = list(dict.fromkeys(sources))

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
        logger.error(
            'ask | mode=decode | cache save FAILED for id=%s: %s', document_id, exc
        )

    logger.info('ask | mode=decode | complete | id=%s', document_id)
    return result


async def _handle_analyze(request: AskRequest) -> AnalysisResponse:
    """
    Deep notice analysis with analysis-specific caching.

    Checks analysis_result cache first. If regenerate=False and cache hit,
    returns immediately. Otherwise runs the full analysis pipeline.
    """
    document_id = request.document_id
    assert document_id is not None  # noqa: S101

    logger.info(
        'ask | mode=analyze | id=%s regenerate=%s', document_id, request.regenerate
    )

    # 1. Check analysis cache
    if not request.regenerate:
        try:
            cached_analysis = await _run(get_analysis_cache, document_id)
        except RuntimeError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={'stage': 'cache', 'error': str(exc)},
            ) from exc

        if cached_analysis is not None:
            logger.info('ask | mode=analyze | cache HIT | id=%s', document_id)
            return AnalysisResponse(**cached_analysis)

    # 2. Resolve extracted text (cache → textract → pre-supplied)
    try:
        cached_doc = await _run(get_cached_doc, document_id)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'cache', 'error': str(exc)},
        ) from exc

    extracted_text = await _resolve_extracted_text(request, cached_doc, document_id)

    # 3. RAG retrieval
    try:
        passages, sources = await _run(retrieve_relevant_law, extracted_text)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'knowledge_base', 'error': str(exc)},
        ) from exc

    retrieved_law = _PASSAGE_SEPARATOR.join(passages)
    unique_sources = list(dict.fromkeys(sources))

    # 4. Generate analysis
    try:
        result: AnalysisResponse = await _run(
            generate_analysis,
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

    # 5. Cache analysis result (non-fatal on failure)
    try:
        await _run(save_analysis_cache, document_id, result.model_dump())
    except RuntimeError as exc:
        logger.error(
            'ask | mode=analyze | cache save FAILED for id=%s: %s', document_id, exc
        )

    # 6. Append to chat history
    _append_history_safe(document_id, 'assistant', 'analyze', result.summary)

    logger.info('ask | mode=analyze | complete | id=%s', document_id)
    return result


async def _handle_strategy(request: AskRequest) -> StrategyResponse:
    """
    Generate a defence strategy grounded in RAG + bounded chat history.
    Personalised when account_details are provided; general otherwise.
    """
    document_id = request.document_id
    assert document_id is not None  # noqa: S101

    logger.info(
        'ask | mode=strategy | id=%s has_account_details=%s',
        document_id,
        bool(request.account_details),
    )

    # 1. Fetch extracted_text from cache (must have been analyzed/decoded first)
    try:
        cached_doc = await _run(get_cached_doc, document_id)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'cache', 'error': str(exc)},
        ) from exc

    extracted_text = _get_extracted_text_from_cache(cached_doc, document_id, 'strategy')

    # 2. Fetch bounded chat history
    try:
        chat_history = await _run(get_chat_history, document_id)
    except RuntimeError as exc:
        logger.warning(
            'ask | mode=strategy | history fetch failed for %s: %s', document_id, exc
        )
        chat_history = []

    # 3. RAG retrieval
    try:
        passages, sources = await _run(retrieve_relevant_law, extracted_text)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'knowledge_base', 'error': str(exc)},
        ) from exc

    retrieved_law = _PASSAGE_SEPARATOR.join(passages)

    # 4. Generate strategy
    try:
        result: StrategyResponse = await _run(
            generate_strategy,
            document_id,
            extracted_text,
            retrieved_law,
            chat_history,
            request.account_details,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={'stage': 'llm_generation', 'error': str(exc)},
        ) from exc

    # 5. Append to chat history
    _append_history_safe(document_id, 'assistant', 'strategy', result.strategy)

    logger.info('ask | mode=strategy | complete | id=%s', document_id)
    return result


async def _handle_draft_html(request: AskRequest) -> DraftHtmlResponse:
    """
    Generate an HTML-formatted formal draft reply to the GST Department.
    Uses the full bounded chat history as context for a coherent draft.
    """
    document_id = request.document_id
    assert document_id is not None  # noqa: S101

    logger.info('ask | mode=draft | id=%s', document_id)

    # 1. Fetch extracted_text from cache
    try:
        cached_doc = await _run(get_cached_doc, document_id)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'cache', 'error': str(exc)},
        ) from exc

    extracted_text = _get_extracted_text_from_cache(cached_doc, document_id, 'draft')

    # 2. Fetch bounded chat history
    try:
        chat_history = await _run(get_chat_history, document_id)
    except RuntimeError as exc:
        logger.warning(
            'ask | mode=draft | history fetch failed for %s: %s', document_id, exc
        )
        chat_history = []

    # 3. RAG retrieval
    try:
        passages, sources = await _run(retrieve_relevant_law, extracted_text)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'knowledge_base', 'error': str(exc)},
        ) from exc

    retrieved_law = _PASSAGE_SEPARATOR.join(passages)
    unique_sources = list(dict.fromkeys(sources))

    # 4. Generate HTML draft
    try:
        result: DraftHtmlResponse = await _run(
            generate_html_draft,
            document_id,
            extracted_text,
            retrieved_law,
            chat_history,
            unique_sources,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={'stage': 'llm_generation', 'error': str(exc)},
        ) from exc

    # 5. Append to chat history (store a plain summary, not the full HTML)
    _append_history_safe(
        document_id,
        'assistant',
        'draft',
        f'HTML draft generated. Citations: {result.citations}',
    )

    logger.info('ask | mode=draft | complete | id=%s', document_id)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _resolve_extracted_text(
    request: AskRequest,
    cached: dict | None,
    document_id: str,
) -> str:
    """
    Resolve the extracted text for a document using the priority chain:
      cached extracted_text → Textract (S3) → pre-supplied extracted_text in request.
    """
    if cached is not None:
        logger.info(
            'ask | using cached extracted_text for id=%s (regenerate=%s)',
            document_id,
            request.regenerate,
        )
        return cached['extracted_text']

    if request.s3_bucket and request.s3_key:
        try:
            return await _run(extract_text_from_s3, request.s3_bucket, request.s3_key)
        except RuntimeError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={'stage': 'textract', 'error': str(exc)},
            ) from exc

    if request.extracted_text and request.extracted_text.strip():
        logger.info('ask | using pre-supplied extracted_text | id=%s', document_id)
        return request.extracted_text

    raise HTTPException(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            'stage': 'textract',
            'error': 'No cached text, S3 coordinates, or extracted_text provided.',
        },
    )


def _get_extracted_text_from_cache(
    cached_doc: dict | None,
    document_id: str,
    mode: str,
) -> str:
    """
    Return cached extracted_text. Raises 422 if no prior decode/analyze was run.
    strategy and draft modes require the document to have been processed first.
    """
    if cached_doc is None or not cached_doc.get('extracted_text'):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                'stage': 'cache',
                'error': (
                    f'No cached document found for document_id={document_id}. '
                    f'Run analyze or decode mode first before requesting {mode}.'
                ),
            },
        )
    return cached_doc['extracted_text']


def _append_history_safe(
    document_id: str,
    role: str,
    mode: str,
    content: str,
) -> None:
    '''
    Fire-and-forget: schedule a history append without blocking the response.
    Uses asyncio.create_task so the caller is never delayed.
    Non-fatal — logs on failure but never surfaces to the caller.
    '''

    async def _fire() -> None:
        try:
            await _run(append_message, document_id, role, mode, content)
        except Exception as exc:
            logger.warning(
                'ask | history append FAILED for id=%s mode=%s: %s', document_id, mode, exc
            )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_fire())
    except RuntimeError as exc:
        logger.warning('ask | could not schedule history append for %s: %s', document_id, exc)

