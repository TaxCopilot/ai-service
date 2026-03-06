import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import BedrockEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy.exc import SQLAlchemyError

from config import settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = 'tax_laws'


def retrieve_relevant_law(query: str, top_k: int = 5) -> tuple[list[str], list[str]]:
    """
    Retrieves the most relevant legal texts from PGVector based on the query.
    Returns a tuple of (passages, sources).
    
    If the database or AWS Bedrock is unavailable, returns ([], []) to 
    trigger a safe 'insufficient information' fallback in the LLM generative layer.
    """
    logger.info('RAG: Searching legal corpus for: %s...', query[:50])

    try:
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        embeddings = BedrockEmbeddings(
            client=bedrock_client,
            model_id='amazon.titan-embed-text-v2:0',
        )

        db_url = settings.database_url
        if db_url.startswith('postgresql://'):
            db_url = db_url.replace('postgresql://', 'postgresql+psycopg://')

        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=_COLLECTION_NAME,
            connection=db_url,
            use_jsonb=True,
        )

        search_results = vector_store.similarity_search_with_score(query, k=min(top_k, 5))
    except (BotoCoreError, ClientError) as exc:
        logger.error('KB API Error: Bedrock embedding generation failed — %s', exc)
        return [], []
    except SQLAlchemyError as exc:
        logger.error('KB DB Error: PGVector similarity search failed — %s', exc)
        return [], []

    if not search_results:
        logger.warning('No relevant legal passages found in DB. Returning empty context.')
        return [], []

    passages = []
    sources = []
    for doc, score in search_results:
        meta = doc.metadata
        section_num = meta.get('section_number')
        section_title = meta.get('section_title')
        source = meta.get('source', 'Unknown Document')
        
        logger.debug('Score: %.4f | Source: %s | Section: %s', score, source, section_num)

        if section_num:
            header = f'[Section {section_num}'
            if section_title:
                header += f': {section_title}'
            header += f' | Source: {source}]'
        else:
            header = f'[Source: {source}]'
            
        passages.append(f'{header}\n{doc.page_content}')
        sources.append(source)

    logger.info('Retrieved %d grounded legal passages.', len(search_results))
    return passages, sources
