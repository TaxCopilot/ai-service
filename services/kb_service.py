import logging
import os

from langchain_aws import BedrockEmbeddings
from langchain_postgres.vectorstores import PGVector

from services.config import settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = 'gst_laws'


def retrieve_relevant_law(query: str, top_k: int = 5) -> str:
    '''
    Retrieves the most relevant legal texts from PGVector based on the query.
    Returns a formatted string containing the relevant sections.
    '''
    print(f'🔍 GST RAG: Searching legal corpus for: {query[:50]}...')

    embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')

    db_url = settings.database_url
    if db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+psycopg://')

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=_COLLECTION_NAME,
        connection=db_url,
        use_jsonb=True,
    )

    search_results = vector_store.similarity_search_with_score(
        query,
        k=min(top_k, 5),
        filter={'tax_type': 'GST'}
    )

    if not search_results:
        print('⚠️ No relevant legal passages found.')
        return ''

    # 4. Format the output with clear section headers for the LLM
    print('--- HACKATHON DEBUG: RETRIEVAL SCORES ---')
    passages = []
    for doc, score in search_results:
        meta = doc.metadata
        section_num = meta.get('section_number')
        section_title = meta.get('section_title')
        
        # Log score for threshold observation
        # IF cosine distance: lower is better (0.0 is exact match)
        # IF cosine similarity: higher is better (1.0 is exact match)
        print(f'Score: {score:.4f} | Section: {section_num}')

        # Create a header like [Section 73: Determination of tax]
        header = f'[Section {section_num}'
        if section_title:
             header += f': {section_title}'
        header += ']'

        passages.append(f'{header}\n{doc.page_content}')

    # Join the passages into a single block of structured context
    law_context = '\n\n'.join(passages)
    print(f'✅ Retrieved {len(search_results)} grounded legal passages.')
    
    return law_context
