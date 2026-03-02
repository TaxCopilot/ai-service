import logging
import os

from langchain_aws import BedrockEmbeddings
from langchain_postgres.vectorstores import PGVector

from config import settings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = 'tax_laws'


def retrieve_relevant_law(query: str, top_k: int = 5) -> str:
    
    #Retrieves the most relevant legal texts from PGVector based on the query.
    #Returns a formatted string containing the relevant sections.
    
    print(f'🔍 GST RAG: Searching legal corpus for: {query[:50]}...')

    import boto3
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id='amazon.titan-embed-text-v2:0'
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

    search_results = vector_store.similarity_search_with_score(
        query,
        k=min(top_k, 5)
    )

    if not search_results:
        print('No relevant legal passages found in DB. Falling back to Mock RAG Data for local testing...')
        mock_passage = (
            "[Section 73: Determination of tax not paid or short paid]\n"
            "(1) Where it appears to the proper officer that any tax has not been paid or short paid "
            "or erroneously refunded, or where input tax credit has been wrongly availed or utilised "
            "for any reason, other than the reason of fraud or any wilful-misstatement or suppression of "
            "facts to evade tax, he shall serve notice on the person chargeable with tax which has not "
            "been so paid or which has been so short paid."
        )
        return mock_passage

    # 4. Format the output with clear section headers for the LLM
    print('--- HACKATHON DEBUG: RETRIEVAL SCORES ---')
    passages = []
    for doc, score in search_results:
        meta = doc.metadata
        section_num = meta.get('section_number')
        section_title = meta.get('section_title')
        source = meta.get('source', 'Unknown Document')
        
        # Log score for threshold observation
        print(f'Score: {score:.4f} | Source: {source} | Section: {section_num}')

        if section_num:
            header = f'[Section {section_num}'
            if section_title:
                header += f': {section_title}'
            header += f' | Source: {source}]'
        else:
            header = f'[Source: {source}]'
            
        passages.append(f'{header}\n{doc.page_content}')

    # Join the passages into a single block of structured context
    law_context = '\n\n'.join(passages)
    print(f'✅ Retrieved {len(search_results)} grounded legal passages.')
    
    return law_context
