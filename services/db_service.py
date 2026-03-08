"""
Postgres document cache and chat session service.

Tables managed here:
  - document_cache  : caches Textract output, draft replies, and analysis results
                      so expensive LLM calls are never repeated for the same document.
  - chat_messages   : persistent per-document chat history (scoped by document_id).
                      History is bounded by max_chars to guard context-window limits.

The tables are created automatically on first use via ensure_table().
"""

import json
import logging
from datetime import datetime, timezone
from typing import TypedDict

import psycopg

from config import settings

logger = logging.getLogger(__name__)

_MAX_HISTORY_CHARS = 6_000  # token budget guard for injected history

_CREATE_CACHE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_cache (
    document_id     TEXT        PRIMARY KEY,
    s3_key          TEXT,
    extracted_text  TEXT        NOT NULL,
    draft_reply     TEXT        NOT NULL,
    citations       JSONB       NOT NULL DEFAULT '[]',
    is_grounded     BOOLEAN     NOT NULL DEFAULT FALSE,
    analysis_result JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_ADD_ANALYSIS_COLUMN_SQL = """
ALTER TABLE document_cache
    ADD COLUMN IF NOT EXISTS analysis_result JSONB;
"""

_CREATE_CHAT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ai_chat_history (
    id          BIGSERIAL   PRIMARY KEY,
    document_id TEXT        NOT NULL,
    role        TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    mode        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ai_chat_history_doc
    ON ai_chat_history (document_id, created_at);
"""

# Migrate existing chat_messages tables that were created before document_id / mode
# columns were added to the schema.
_MIGRATE_CHAT_TABLE_SQL = """
ALTER TABLE ai_chat_history
    ADD COLUMN IF NOT EXISTS document_id TEXT NOT NULL DEFAULT '';
ALTER TABLE ai_chat_history
    ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'chat';
"""


class CachedDocument(TypedDict):
    document_id: str
    s3_key: str | None
    extracted_text: str
    draft_reply: str
    citations: list[str]
    is_grounded: bool


class ChatMessageRow(TypedDict):
    role: str
    mode: str
    content: str


def _get_conn_str() -> str:
    """Return the raw psycopg connection string, rewriting prefix if needed."""
    url = settings.database_url
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', 'postgresql+psycopg://', 1)
    return url.replace('postgresql+psycopg://', 'postgresql://', 1)


def ensure_table() -> None:
    """
    Create document_cache and chat_messages tables if they do not already exist.
    Also migrates existing document_cache tables to include the analysis_result column.
    Safe to call multiple times (fully idempotent).
    """
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            conn.execute(_CREATE_CACHE_TABLE_SQL)
            conn.execute(_ADD_ANALYSIS_COLUMN_SQL)
            conn.execute(_CREATE_CHAT_TABLE_SQL)
            conn.execute(_MIGRATE_CHAT_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        logger.exception('Failed to ensure DB tables: %s', exc)
        raise RuntimeError(f'DB setup failed: {exc}') from exc
    logger.info('DB tables ensured (document_cache, ai_chat_history).')


# ---------------------------------------------------------------------------
# document_cache helpers
# ---------------------------------------------------------------------------


def get_cached_doc(document_id: str) -> CachedDocument | None:
    """
    Fetch a previously processed document from the cache.
    Returns None if the document has not been processed before.
    """
    sql = """
        SELECT document_id, s3_key, extracted_text, draft_reply, citations, is_grounded
        FROM document_cache
        WHERE document_id = %s
    """
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            row = conn.execute(sql, (document_id,)).fetchone()
    except psycopg.Error as exc:
        logger.exception(
            'DB error fetching cache for document_id=%s: %s', document_id, exc
        )
        raise RuntimeError(f'Cache lookup failed for {document_id}: {exc}') from exc

    if row is None:
        return None

    doc_id, s3_key, extracted_text, draft_reply, citations_raw, is_grounded = row
    citations: list[str] = (
        json.loads(citations_raw)
        if isinstance(citations_raw, str)
        else (citations_raw or [])
    )
    logger.info('Cache HIT for document_id=%s', document_id)
    return CachedDocument(
        document_id=doc_id,
        s3_key=s3_key,
        extracted_text=extracted_text,
        draft_reply=draft_reply,
        citations=citations,
        is_grounded=is_grounded,
    )


def save_cached_doc(
    document_id: str,
    extracted_text: str,
    draft_reply: str,
    citations: list[str],
    is_grounded: bool,
    s3_key: str | None = None,
) -> None:
    """
    Upsert a processed document into the cache.
    Updates the existing row if document_id already exists (e.g., regenerate flow).
    """
    sql = """
        INSERT INTO document_cache
            (document_id, s3_key, extracted_text, draft_reply, citations, is_grounded, updated_at)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            draft_reply   = EXCLUDED.draft_reply,
            citations     = EXCLUDED.citations,
            is_grounded   = EXCLUDED.is_grounded,
            updated_at    = EXCLUDED.updated_at
    """
    now = datetime.now(timezone.utc)
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            conn.execute(
                sql,
                (
                    document_id,
                    s3_key,
                    extracted_text,
                    draft_reply,
                    json.dumps(citations),
                    is_grounded,
                    now,
                ),
            )
            conn.commit()
    except psycopg.Error as exc:
        logger.exception(
            'DB error saving cache for document_id=%s: %s', document_id, exc
        )
        raise RuntimeError(f'Cache save failed for {document_id}: {exc}') from exc

    logger.info('Cache SAVED for document_id=%s', document_id)


def get_analysis_cache(document_id: str) -> dict | None:
    """
    Fetch a cached analysis result for a document.
    Returns None if no analysis has been run yet.
    """
    sql = 'SELECT analysis_result FROM document_cache WHERE document_id = %s'
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            row = conn.execute(sql, (document_id,)).fetchone()
    except psycopg.Error as exc:
        logger.exception(
            'DB error fetching analysis cache for %s: %s', document_id, exc
        )
        raise RuntimeError(
            f'Analysis cache lookup failed for {document_id}: {exc}'
        ) from exc

    if row is None or row[0] is None:
        return None

    raw = row[0]
    result: dict = json.loads(raw) if isinstance(raw, str) else raw
    logger.info('Analysis cache HIT for document_id=%s', document_id)
    return result


def save_analysis_cache(document_id: str, analysis_dict: dict, extracted_text: str = '') -> None:
    """
    Persist an analysis result in the document_cache row.
    Assumes the document_cache row already exists (created by save_cached_doc).
    Uses INSERT ... ON CONFLICT to handle the case where only analysis is being cached
    (no prior decode run).
    """
    sql = """
        INSERT INTO document_cache
            (document_id, extracted_text, draft_reply, citations, analysis_result, updated_at)
        VALUES (%s, %s, '', '[]'::jsonb, %s::jsonb, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            extracted_text  = CASE WHEN EXCLUDED.extracted_text != '' THEN EXCLUDED.extracted_text ELSE document_cache.extracted_text END,
            analysis_result = EXCLUDED.analysis_result,
            updated_at      = EXCLUDED.updated_at
    """
    now = datetime.now(timezone.utc)
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            conn.execute(sql, (document_id, extracted_text, json.dumps(analysis_dict), now))
            conn.commit()
    except psycopg.Error as exc:
        logger.exception('DB error saving analysis cache for %s: %s', document_id, exc)
        raise RuntimeError(
            f'Analysis cache save failed for {document_id}: {exc}'
        ) from exc


def update_extracted_text(document_id: str, extracted_text: str) -> None:
    """
    Just updates or inserts the extracted_text for a document.
    """
    sql = """
        INSERT INTO document_cache
            (document_id, extracted_text, draft_reply, citations, analysis_result, updated_at)
        VALUES (%s, %s, '', '[]'::jsonb, '{}'::jsonb, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            extracted_text = EXCLUDED.extracted_text,
            updated_at     = EXCLUDED.updated_at
    """
    now = datetime.now(timezone.utc)
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            conn.execute(sql, (document_id, extracted_text, now))
            conn.commit()
    except psycopg.Error as exc:
        logger.exception('DB error updating extracted_text for %s: %s', document_id, exc)
        # non-fatal


    logger.info('Analysis cache SAVED for document_id=%s', document_id)


# ---------------------------------------------------------------------------
# ai_chat_history helpers
# ---------------------------------------------------------------------------


def append_message(session_id: str, role: str, mode: str, content: str) -> None:
    """Append a single message to the AI chat history for a session/case."""
    sql = """
        INSERT INTO ai_chat_history (document_id, role, mode, content)
        VALUES (%s, %s, %s, %s)
    """
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            conn.execute(sql, (session_id, role, mode, content))
            conn.commit()
    except psycopg.Error as exc:
        logger.exception('DB error appending message for %s: %s', session_id, exc)
        raise RuntimeError(f'Message append failed for {session_id}: {exc}') from exc


def get_chat_history(session_id: str) -> list[ChatMessageRow]:
    """
    Fetch the bounded chat history for a session, ordered chronologically.

    Fetches the most recent messages first, accumulates up to _MAX_HISTORY_CHARS,
    then reverses to chronological order. This ensures the injected context never
    exceeds the LLM context-window budget.
    """
    sql = """
        SELECT role, mode, content
        FROM ai_chat_history
        WHERE document_id = %s
        ORDER BY created_at DESC
    """
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            rows = conn.execute(sql, (session_id,)).fetchall()
    except psycopg.Error as exc:
        logger.exception('DB error fetching history for %s: %s', session_id, exc)
        raise RuntimeError(f'History fetch failed for {session_id}: {exc}') from exc

    accumulated = 0
    bounded: list[ChatMessageRow] = []
    for role, mode, content in rows:
        if accumulated + len(content) > _MAX_HISTORY_CHARS:
            break
        bounded.append(ChatMessageRow(role=role, mode=mode, content=content))
        accumulated += len(content)

    # Reverse to restore chronological order
    bounded.reverse()
    logger.info(
        'History fetched for session_id=%s: %d messages (%d chars)',
        session_id,
        len(bounded),
        accumulated,
    )
    return bounded
