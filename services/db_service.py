'''
Postgres document cache service.

Provides get / save helpers for the `document_cache` table so that expensive
Textract OCR and Gemini calls are never repeated for the same document_id.

The table is created automatically on first use — no migration script needed.
'''

import json
import logging
from datetime import datetime, timezone
from typing import TypedDict

import psycopg

from config import settings

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS document_cache (
    document_id   TEXT        PRIMARY KEY,
    s3_key        TEXT,
    extracted_text TEXT       NOT NULL,
    draft_reply   TEXT        NOT NULL,
    citations     JSONB       NOT NULL DEFAULT '[]',
    is_grounded   BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
'''


class CachedDocument(TypedDict):
    document_id: str
    s3_key: str | None
    extracted_text: str
    draft_reply: str
    citations: list[str]
    is_grounded: bool


def _get_conn_str() -> str:
    '''Return the raw psycopg connection string, rewriting prefix if needed.'''
    url = settings.database_url
    if url.startswith('postgresql://'):
        url = url.replace('postgresql://', 'postgresql+psycopg://', 1)
    # psycopg expects plain postgresql:// not the SQLAlchemy dialect prefix
    return url.replace('postgresql+psycopg://', 'postgresql://', 1)


def ensure_table() -> None:
    '''Create the document_cache table if it does not already exist.'''
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        logger.exception('Failed to ensure document_cache table: %s', exc)
        raise RuntimeError(f'DB setup failed: {exc}') from exc
    logger.info('document_cache table ensured.')


def get_cached_doc(document_id: str) -> CachedDocument | None:
    '''
    Fetch a previously processed document from the cache.

    Returns None if the document has not been processed before.
    '''
    sql = '''
        SELECT document_id, s3_key, extracted_text, draft_reply, citations, is_grounded
        FROM document_cache
        WHERE document_id = %s
    '''
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            row = conn.execute(sql, (document_id,)).fetchone()
    except psycopg.Error as exc:
        logger.exception('DB error fetching cache for document_id=%s: %s', document_id, exc)
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
    '''
    Upsert a processed document into the cache.

    Updates the existing row if document_id already exists (e.g., regenerate flow).
    '''
    sql = '''
        INSERT INTO document_cache
            (document_id, s3_key, extracted_text, draft_reply, citations, is_grounded, updated_at)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            draft_reply   = EXCLUDED.draft_reply,
            citations     = EXCLUDED.citations,
            is_grounded   = EXCLUDED.is_grounded,
            updated_at    = EXCLUDED.updated_at
    '''
    now = datetime.now(timezone.utc)
    try:
        with psycopg.connect(_get_conn_str()) as conn:
            conn.execute(sql, (
                document_id,
                s3_key,
                extracted_text,
                draft_reply,
                json.dumps(citations),
                is_grounded,
                now,
            ))
            conn.commit()
    except psycopg.Error as exc:
        logger.exception('DB error saving cache for document_id=%s: %s', document_id, exc)
        raise RuntimeError(f'Cache save failed for {document_id}: {exc}') from exc

    logger.info('Cache SAVED for document_id=%s', document_id)
