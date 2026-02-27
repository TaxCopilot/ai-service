"""
Retrieves relevant tax law passages from a Bedrock Knowledge Base.

Upload law PDFs to S3 and sync the KB from the AWS Console — no ingestion
script needed. AWS handles chunking, embedding, and vector indexing.

Required IAM: bedrock:Retrieve on the knowledge base resource.
"""

import logging

import boto3
import botocore.client
from botocore.exceptions import BotoCoreError, ClientError

from config import settings

logger = logging.getLogger(__name__)

_BEDROCK_AGENT_SERVICE = 'bedrock-agent-runtime'

_kb_client: botocore.client.BaseClient | None = None


def _get_client() -> botocore.client.BaseClient:
    global _kb_client
    if _kb_client is None:
        _kb_client = boto3.client(_BEDROCK_AGENT_SERVICE, region_name=settings.aws_region)
    return _kb_client


def retrieve_relevant_law(query: str) -> tuple[list[str], list[str]]:
    """
    Run a semantic search against the tax law Knowledge Base.

    Returns (passages, sources) — both lists are ordered by relevance score
    and will be empty if the KB has no matching content.
    """
    client = _get_client()

    logger.info(
        'KB query | kb=%s top_k=%d',
        settings.bedrock_knowledge_base_id,
        settings.bedrock_retrieval_results,
    )

    try:
        response = client.retrieve(
            knowledgeBaseId=settings.bedrock_knowledge_base_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': settings.bedrock_retrieval_results,
                }
            },
        )
    except ClientError as exc:
        code = exc.response['Error']['Code']
        logger.exception('Bedrock KB error: %s', code)
        raise RuntimeError(
            f"Knowledge Base retrieval failed ({code}). "
            f'Check BEDROCK_KNOWLEDGE_BASE_ID and IAM permissions.'
        ) from exc
    except BotoCoreError as exc:
        logger.exception('Bedrock KB connection error')
        raise RuntimeError(f'Knowledge Base connection error: {exc}') from exc

    results: list[dict] = response.get('retrievalResults', [])
    if not results:
        logger.warning('KB returned no results for query: %.100s', query)
        return [], []

    passages = [r['content']['text'] for r in results]
    sources = [
        r.get('location', {}).get('s3Location', {}).get('uri', 'unknown')
        for r in results
    ]

    logger.info('KB returned %d passages', len(passages))
    return passages, sources
