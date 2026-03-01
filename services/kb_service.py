"""
Retrieves relevant tax law passages from the pgvector database.

Tax law PDFs are pre-chunked and embedded via scripts/ingest_to_pgvector.py.
This module runs a cosine-similarity search against those embeddings to
find the most relevant passages for a given query.

Required env: DATABASE_URL, AWS credentials for Bedrock Titan embeddings.
"""

import logging

from langchain_aws import BedrockEmbeddings
from langchain_postgres.vectorstores import PGVector

from config import settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = 'tax_laws'
_EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'

_vector_store: PGVector | None = None


def _get_vector_store() -> PGVector:
    """Lazily initialise the PGVector connection on first call."""
    global _vector_store
    if _vector_store is None:
        db_url = settings.database_url
        if db_url.startswith('postgresql://'):
            db_url = db_url.replace('postgresql://', 'postgresql+psycopg://')

        import boto3
        import botocore.config

        boto_config = botocore.config.Config(
            region_name=settings.aws_region,
            retries={'max_attempts': 3, 'mode': 'standard'}
        )
        
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=boto_config,
        )

        embeddings = BedrockEmbeddings(
            client=bedrock_client,
            model_id=_EMBEDDING_MODEL_ID,
            region_name=settings.aws_region,
        )

        _vector_store = PGVector(
            embeddings=embeddings,
            collection_name=_COLLECTION_NAME,
            connection=db_url,
            use_jsonb=True,
        )
        logger.info(
            'pgvector store initialised | collection=%s',
            _COLLECTION_NAME,
        )
    return _vector_store


def retrieve_relevant_law(query: str) -> tuple[list[str], list[str]]:
    """
    Run a semantic search against the tax law vector database.

    Returns (passages, sources) — both lists are ordered by relevance score
    and will be empty if the database has no matching content.
    """
    top_k = settings.bedrock_retrieval_results

    logger.info(
        'pgvector query | collection=%s top_k=%d',
        _COLLECTION_NAME,
        top_k,
    )

    try:
        store = _get_vector_store()
        results = store.similarity_search(query, k=top_k)
    except Exception as exc:
        logger.exception('pgvector retrieval failed')
        raise RuntimeError(
            f'Knowledge Base retrieval failed: {exc}. '
            'Check DATABASE_URL and AWS credentials.'
        ) from exc

    if not results:
        logger.warning('pgvector returned no results for query: %.100s', query)
        return [], []

    passages = [doc.page_content for doc in results]
    sources = [doc.metadata.get('source', 'unknown') for doc in results]

    logger.info('pgvector returned %d passages', len(passages))
    return passages, sources
