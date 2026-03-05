'''
Utility script: print the total number of chunks stored in the tax_laws
vector store. Useful for monitoring the state of the database.
'''

import asyncio

import boto3
from langchain_aws import BedrockEmbeddings
from langchain_postgres.vectorstores import PGVector

from config import settings

_COLLECTION_NAME = 'tax_laws'
_EMBEDDING_MODEL = 'amazon.titan-embed-text-v2:0'
_COUNT_SQL = (
    'SELECT count(*) FROM langchain_pg_embedding '
    "WHERE collection_id = ("
    '    SELECT uuid FROM langchain_pg_collection WHERE name = :name'
    ')'
)


async def check() -> None:
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=_EMBEDDING_MODEL,
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

    with vector_store._make_sync_session() as session:
        from sqlalchemy import text

        result = session.execute(
            text(_COUNT_SQL),
            {'name': _COLLECTION_NAME},
        )
        count = result.scalar()
        print(f'Total documents in {_COLLECTION_NAME}: {count}')


if __name__ == '__main__':
    asyncio.run(check())
