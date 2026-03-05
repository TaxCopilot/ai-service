'''
Utility script: truncate all vector store tables for a clean re-ingestion.

WARNING: This permanently deletes all embedded documents. Use only in
development or when restarting the ingestion from scratch.
'''

import psycopg

from config import settings


def clean_db() -> None:
    db_url = settings.database_url.replace('postgresql+psycopg://', 'postgresql://')
    print('Cleaning langchain_pg_embedding and langchain_pg_collection tables...')
    with psycopg.connect(db_url) as conn:
        conn.execute('TRUNCATE TABLE langchain_pg_embedding CASCADE')
        conn.execute('TRUNCATE TABLE langchain_pg_collection CASCADE')
        conn.commit()
    print('Done! Database is now empty.')


if __name__ == '__main__':
    clean_db()
