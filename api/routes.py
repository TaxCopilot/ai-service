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
    update_extracted_text,
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


class DocumentRef(BaseModel):
    """A reference to a single tax document for processing."""
    document_id: str
    s3_bucket: str | None = None
    s3_key: str | None = None
    filename: str | None = None
    extracted_text: str | None = Field(
        default=None,
        description='Pre-extracted notice text. Skips Textract when provided.',
    )


class AskRequest(BaseModel):
    """
    Unified request for all AI modes.

    chat     — set mode="chat" and provide a message.
    decode   — set mode="decode" and provide s3_bucket + s3_key (or extracted_text).
    analyze  — set mode="analyze" and provide document details.
    strategy — set mode="strategy" and provide documents (+ optional account_details).
    draft    — set mode="draft" and provide documents.
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

    # --- multi-doc & session fields ---
    documents: list[DocumentRef] | None = Field(
        default=None,
        description='List of documents to process in this request.',
    )
    session_id: str | None = Field(
        default=None,
        description='Optional unified session ID (e.g., caseId) grouping these documents for chat history.',
    )

    # --- legacy shared document fields (backward compatibility) ---
    document_id: str | None = Field(
        default=None,
        description='Legacy field. Use `documents` instead.',
    )
    s3_bucket: str | None = Field(default=None)
    s3_key: str | None = Field(default=None)
    extracted_text: str | None = Field(default=None)
    notice_type: str | None = Field(default=None)

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
    """Validate mode-specific required fields and normalize inputs."""
    # Normalize legacy single-doc fields into the `documents` array
    if req.document_id and not req.documents:
        req.documents = [
            DocumentRef(
                document_id=req.document_id,
                s3_bucket=req.s3_bucket,
                s3_key=req.s3_key,
                extracted_text=req.extracted_text,
            )
        ]
        
    # If no session_id is provided, fallback to the first document_id
    if not req.session_id and req.documents and len(req.documents) > 0:
        req.session_id = req.documents[0].document_id

    if req.mode == 'chat':
        if not req.message or not req.message.strip():
            raise ValueError('message is required for chat mode.')

    elif req.mode in ('decode', 'analyze', 'strategy', 'draft'):
        if not req.documents or len(req.documents) == 0:
            raise ValueError(f'At least one document is required for {req.mode} mode.')
        
        if req.mode == 'decode':
            for doc in req.documents:
                _require_text_source(doc)

    return req


def _require_text_source(doc: DocumentRef) -> None:
    """Raises ValueError if neither s3 coords nor extracted_text are provided for a doc."""
    has_s3 = bool(doc.s3_bucket and doc.s3_key)
    has_text = bool(doc.extracted_text and doc.extracted_text.strip())
    if not has_s3 and not has_text:
        raise ValueError(
            f'Document {doc.document_id} requires either (s3_bucket + s3_key) or extracted_text.'
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
    assert request.documents is not None
    assert request.session_id is not None

    logger.info("ask | mode=chat | q='%s...'", message[:60])

    extracted_texts = []
    chat_history = []
    
    for doc in request.documents:
        try:
            cached_doc = await _run(get_cached_doc, doc.document_id)
            if cached_doc and cached_doc.get('extracted_text'):
                extracted_texts.append(f"--- Document: {doc.filename or doc.document_id} ---\n{cached_doc['extracted_text']}")
        except RuntimeError as exc:
            logger.warning('ask | mode=chat | doc context fetch failed for %s: %s', doc.document_id, exc)

    combined_extracted_text = _PASSAGE_SEPARATOR.join(extracted_texts) if extracted_texts else None

    try:
        chat_history = await _run(get_chat_history, request.session_id)
    except RuntimeError as exc:
        logger.warning('ask | mode=chat | history fetch failed for %s: %s', request.session_id, exc)

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
            combined_extracted_text,
            chat_history,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={'stage': 'llm_generation', 'error': str(exc)},
        ) from exc

    _append_history_safe(request.session_id, 'user', 'chat', message)
    _append_history_safe(request.session_id, 'assistant', 'chat', result.answer)

    logger.info('ask | mode=chat | complete')
    return result


async def _handle_decode(request: AskRequest) -> NoticeResponse:
    """
    Full decode pipeline with persistent Postgres caching.

    Extracts text for all documents provided, but currently only returns a single 
    NoticeResponse (representing the primary/first document).
    """
    assert request.documents is not None
    assert request.session_id is not None
    
    # We use the primary document for decode caching/response
    primary_doc = request.documents[0]
    document_id = primary_doc.document_id

    logger.info(
        'ask | mode=decode | session_id=%s regenerate=%s primary_doc=%s',
        request.session_id,
        request.regenerate,
        document_id,
    )

    try:
        cached = await _run(get_cached_doc, document_id)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={'stage': 'cache', 'error': str(exc)},
        ) from exc

    extracted_text = await _resolve_doc_text(primary_doc, cached, request.regenerate)

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
            primary_doc.s3_key,
        )
    except RuntimeError as exc:
        logger.error(
            'ask | mode=decode | cache save FAILED for id=%s: %s', document_id, exc
        )

    logger.info('ask | mode=decode | complete | id=%s', document_id)
    return result


async def _handle_analyze(request: AskRequest) -> AnalysisResponse:
    """
    Deep notice analysis representing multiple documents.
    """
    assert request.documents is not None
    assert request.session_id is not None

    logger.info(
        'ask | mode=analyze | session_id=%s regenerate=%s docs=%d', 
        request.session_id, request.regenerate, len(request.documents)
    )

    # Resolve text for all documents
    extracted_texts_list: list[tuple[str, str]] = []
    combined_text_for_rag = []
    
    for doc in request.documents:
        try:
            cached_doc = await _run(get_cached_doc, doc.document_id)
        except RuntimeError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={'stage': 'cache', 'error': str(exc)},
            ) from exc

        text = await _resolve_doc_text(doc, cached_doc, request.regenerate)
        filename = doc.filename or f'Document {doc.document_id}.pdf'
        extracted_texts_list.append((text, filename))
        combined_text_for_rag.append(text)

    # 3. RAG retrieval using combined text
    combined_extracted = _PASSAGE_SEPARATOR.join(combined_text_for_rag)
    try:
        passages, sources = await _run(retrieve_relevant_law, combined_extracted)
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
            request.session_id,
            extracted_texts_list,
            retrieved_law,
            unique_sources,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={'stage': 'llm_generation', 'error': str(exc)},
        ) from exc

    # 5. Cache analysis result per-document (just linking the analysis to each doc ID)
    for doc, (text, _) in zip(request.documents, extracted_texts_list):
        try:
            await _run(save_analysis_cache, doc.document_id, result.model_dump(), text)
        except RuntimeError as exc:
            logger.error(
                'ask | mode=analyze | cache save FAILED for id=%s: %s', doc.document_id, exc
            )

    # 6. Append to chat history — use the full report
    _append_history_safe(request.session_id, 'assistant', 'analyze', result.report)

    logger.info('ask | mode=analyze | complete | session_id=%s', request.session_id)
    return result


async def _handle_strategy(request: AskRequest) -> StrategyResponse:
    """
    Generate a defence strategy grounded in RAG + bounded chat history.
    Personalised when account_details are provided; general otherwise.
    """
    assert request.documents is not None
    assert request.session_id is not None

    logger.info(
        'ask | mode=strategy | session_id=%s has_account_details=%s docs=%d',
        request.session_id,
        bool(request.account_details),
        len(request.documents),
    )

    extracted_texts_list = []
    
    # 1. Fetch extracted_text from cache for all docs (must have been analyzed/decoded first)
    for doc in request.documents:
        try:
            cached_doc = await _run(get_cached_doc, doc.document_id)
        except RuntimeError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={'stage': 'cache', 'error': str(exc)},
            ) from exc

        text = await _resolve_doc_text(doc, cached_doc, getattr(request, 'regenerate', False))
        extracted_texts_list.append(f"--- Document: {doc.filename or doc.document_id} ---\n{text}")

    combined_extracted_text = _PASSAGE_SEPARATOR.join(extracted_texts_list)

    # 2. Fetch bounded chat history
    try:
        chat_history = await _run(get_chat_history, request.session_id)
    except RuntimeError as exc:
        logger.warning(
            'ask | mode=strategy | history fetch failed for %s: %s', request.session_id, exc
        )
        chat_history = []

    # 3. RAG retrieval
    try:
        passages, sources = await _run(retrieve_relevant_law, combined_extracted_text)
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
            request.session_id,
            combined_extracted_text,
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
    _append_history_safe(request.session_id, 'assistant', 'strategy', result.strategy)

    logger.info('ask | mode=strategy | complete | session_id=%s', request.session_id)
    return result


async def _handle_draft_html(request: AskRequest) -> DraftHtmlResponse:
    """
    Generate an HTML-formatted formal draft reply to the GST Department.
    Uses the full bounded chat history as context for a coherent draft.
    """
    assert request.documents is not None
    assert request.session_id is not None

    logger.info('ask | mode=draft | session_id=%s docs=%d', request.session_id, len(request.documents))

    extracted_texts_list = []
    
    # 1. Fetch extracted_text from cache for all docs
    for doc in request.documents:
        try:
            cached_doc = await _run(get_cached_doc, doc.document_id)
        except RuntimeError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={'stage': 'cache', 'error': str(exc)},
            ) from exc

        text = await _resolve_doc_text(doc, cached_doc, getattr(request, 'regenerate', False))
        extracted_texts_list.append(f"--- Document: {doc.filename or doc.document_id} ---\n{text}")

    combined_extracted_text = _PASSAGE_SEPARATOR.join(extracted_texts_list)

    # 2. Fetch bounded chat history
    try:
        chat_history = await _run(get_chat_history, request.session_id)
    except RuntimeError as exc:
        logger.warning(
            'ask | mode=draft | history fetch failed for %s: %s', request.session_id, exc
        )
        chat_history = []

    # 3. RAG retrieval using combined text
    try:
        passages, sources = await _run(retrieve_relevant_law, combined_extracted_text)
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
            request.session_id,
            combined_extracted_text,
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
        request.session_id,
        'assistant',
        'draft',
        f'HTML draft generated. Citations: {result.citations}',
    )

    logger.info('ask | mode=draft | complete | session_id=%s', request.session_id)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _resolve_doc_text(
    doc: DocumentRef,
    cached: dict | None,
    regenerate: bool,
) -> str:
    """
    Resolve the extracted text for a document using the priority chain:
      cached extracted_text → Textract (S3) → pre-supplied extracted_text.
    """
    if cached is not None and cached.get('extracted_text') and not regenerate:
        logger.info(
            'ask | using cached extracted_text for id=%s (regenerate=False)',
            doc.document_id,
        )
        return cached['extracted_text']

    if doc.s3_bucket and doc.s3_key:
        try:
            text = await _run(extract_text_from_s3, doc.s3_bucket, doc.s3_key)
            # CACHE the text so subsequent calls don't hit textract
            try:
                await _run(update_extracted_text, doc.document_id, text)
            except Exception as e:
                logger.warning("ask | failed to cache extracted text for %s: %s", doc.document_id, e)
            return text
        except RuntimeError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={'stage': 'textract', 'error': str(exc)},
            ) from exc

    if doc.extracted_text and doc.extracted_text.strip():
        logger.info('ask | using pre-supplied extracted_text | id=%s', doc.document_id)
        return doc.extracted_text

    raise HTTPException(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            'stage': 'textract',
            'error': f'No valid source (cache/s3/text) provided for doc {doc.document_id}.',
        },
    )


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

