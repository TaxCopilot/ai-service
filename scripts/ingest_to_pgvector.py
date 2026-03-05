import os
import re
import time
from datetime import datetime
from pathlib import Path

import boto3
import fitz
from botocore.client import BaseClient
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

ENV_PATH = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=ENV_PATH)

DATA_DIR = Path(__file__).parent.parent / 'data'
SAFE_BATCH_SIZE = 5
BEDROCK_MODEL_ID = 'amazon.titan-embed-text-v2:0'
COLLECTION_NAME = 'tax_laws'
SUPPORTED_DOC_TYPES = ['act', 'rule', 'notification', 'circular']

DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg://')

if not DATABASE_URL:
    raise ValueError('DATABASE_URL is missing from environment variables')


def extract_doc_number(document_type: str, filename: str) -> str | None:
    if document_type == 'notification':
        match = re.search(r'(\d+)-(\d{4})', filename)
        if match:
            return f'{match.group(1)}/{match.group(2)}'
    elif document_type == 'circular':
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1)
    return None


def extract_issue_date(text: str) -> str | None:
    header_text = text[:1000]
    match = re.search(r'(?i)dated the\s+(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})', header_text)
    if match:
        day, month_str, year = match.groups()
        try:
            date_obj = datetime.strptime(f'{day} {month_str[:3].capitalize()} {year}', '%d %b %Y')
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            return None
    return None


def clean_text(text: str) -> str:
    cleaned = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    headers_to_remove = [r'(?i)Government of India', r'(?i)Ministry of Finance', r'(?i)CBIC']
    for header in headers_to_remove:
        cleaned = re.sub(header, '', cleaned)
        
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    return cleaned.strip()


def extract_document(pdf_path: Path) -> dict | None:
    print(f'📄 Processing: {pdf_path.name}')
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f'❌ Failed to parse {pdf_path.name}: {e}')
        return None

    full_text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text() + '\n'
        
    if not full_text.strip():
        print(f'⚠️ No text extracted from {pdf_path.name}')
        return None

    parent_folder = pdf_path.parent.name.lower()
    doc_type_mapping = {
        'acts': 'act',
        'rules': 'rule',
        'notifications': 'notification',
        'circulars': 'circular'
    }
    document_type = doc_type_mapping.get(parent_folder, 'unknown')
    if document_type not in SUPPORTED_DOC_TYPES:
        document_type = 'unknown'
    
    issue_date = extract_issue_date(full_text)
    doc_number = extract_doc_number(document_type, pdf_path.name)
    cleaned_text = clean_text(full_text)
    
    if not cleaned_text:
        return None

    return {
        'text': cleaned_text,
        'metadata': {
            'document_type': document_type,
            'doc_number': doc_number,
            'issue_date': issue_date,
            'source_filename': pdf_path.name,
            'tax_type': 'GST',
            'section_number': None,
            'section_title': None,
            'status': 'active'
        }
    }


def chunk_document(document: dict) -> list[dict]:
    source_filename = document['metadata']['source_filename']
    print(f'✂️ Chunking: {source_filename}')
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            r'\n(?=Section\s+\d+)',
            r'\n(?=Rule\s+\d+)',
            r'\n(?=CHAPTER)',
            '\n\n',
            '\n',
            ' ',
            ''
        ],
        is_separator_regex=True
    )
    
    chunks = text_splitter.split_text(document['text'])
    
    current_section_number = None
    current_section_title = None
    
    result = []
    for chunk in chunks:
        match = re.search(r'(?i)\n?(?:Section|Rule|CHAPTER)\s+([0-9A-Z]+)[.\-\s]*([^\n]*)', chunk[:200])
        if match:
            current_section_number = match.group(1).strip()
            title = match.group(2).strip()
            current_section_title = title if title else None
            
        base_meta = document['metadata'].copy()
        base_meta['section_number'] = current_section_number
        base_meta['section_title'] = current_section_title
        
        result.append({
            'text': chunk,
            'metadata': base_meta
        })
        
    return result


def get_bedrock_client() -> BaseClient:
    return boto3.client(
        'bedrock-runtime',
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )


def ingest_to_pgvector() -> None:
    print('🚀 Starting ingestion process...')
    
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f'❌ Data directory not found: {DATA_DIR}')
        return

    bedrock_client = get_bedrock_client()
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=BEDROCK_MODEL_ID
    )
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    
    pdf_files = list(DATA_DIR.rglob('*.pdf'))
    if not pdf_files:
        print(f'⚠️ No PDF files found in {DATA_DIR}')
        return

    print(f'📂 Found {len(pdf_files)} PDF files in total.')

    # Get already processed files from the vector store to skip them
    processed_files = set()
    try:
        with vector_store._make_sync_session() as session:
            from sqlalchemy import text
            query = text("SELECT DISTINCT(c.cmetadata->>'source_filename') FROM langchain_pg_embedding c")
            result = session.execute(query)
            processed_files = {row[0] for row in result if row[0]}
            if processed_files:
                print(f'⏭️ Found {len(processed_files)} already processed files in DB. They will be skipped.')
    except Exception as e:
        print(f'⚠️ Could not fetch processed files list: {e}. Proceeding with potential duplicates.')

    all_texts = []
    all_metadatas = []

    for pdf_path in pdf_files:
        if pdf_path.name in processed_files:
            print(f'⏩ Skipping {pdf_path.name} (already in DB)')
            continue

        doc = extract_document(pdf_path)
        if doc:
            chunks = chunk_document(doc)
            for chunk in chunks:
                all_texts.append(chunk['text'])
                all_metadatas.append(chunk['metadata'])
            print(f'✅ Generated {len(chunks)} chunks from {pdf_path.name}')

    total_chunks = len(all_texts)
    if total_chunks == 0:
        print('🎉 No new content to ingest. Everything is already in the database!')
        return

    print(f'💾 Upserting {total_chunks} total chunks in safe batches of {SAFE_BATCH_SIZE}...')
    
    for start in range(0, total_chunks, SAFE_BATCH_SIZE):
        end = min(start + SAFE_BATCH_SIZE, total_chunks)
        batch_texts = all_texts[start:end]
        batch_meta = all_metadatas[start:end]
        
        try:
            vector_store.add_texts(texts=batch_texts, metadatas=batch_meta)
        except Exception as e:
            print(f'⚠️ Embedding batch failed, retrying once. Error: {e}')
            time.sleep(2)
            try:
                vector_store.add_texts(texts=batch_texts, metadatas=batch_meta)
            except Exception as retry_e:
                print(f'❌ Retry failed for batch {start}-{end}. Skipping. Error: {retry_e}')
                continue
            
        print(f'  ✅ Batch {start // SAFE_BATCH_SIZE + 1}/{(total_chunks + SAFE_BATCH_SIZE - 1) // SAFE_BATCH_SIZE} done ({end}/{total_chunks} chunks)')
        time.sleep(2)
        
    print('🎉 Ingestion complete!')


if __name__ == '__main__':
    ingest_to_pgvector()
