import os
import re
from pathlib import Path

from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_aws import BedrockEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables explicitly
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

DATA_DIR = Path(__file__).parent.parent / 'data'
BATCH_SIZE = 50

# LangChain's PGVector needs psycopg plugin for psycopg3
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg://')

# Explicitly use the Titan Embedding v2 model. 
# Do not default to the Claude LLM model id defined in .env
BEDROCK_MODEL_ID = 'amazon.titan-embed-text-v2:0'

if not DATABASE_URL:
    raise ValueError('DATABASE_URL is missing from environment variables')


def extract_text_from_pdf(pdf_path: Path) -> dict:
    '''Extracts text from a PDF file using PyMuPDF and returns a dict with metadata.'''
    print(f'📄 Processing: {pdf_path.name}')
    doc = fitz.open(pdf_path)
    full_text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += text + '\n'
    
    full_text = re.sub(r'\n\d+\n', '\n', full_text)  # remove standalone page numbers
    full_text = re.sub(r'\n{2,}', '\n\n', full_text)
    full_text = re.sub(r'[ \t]+', ' ', full_text)
    
    doc_type = 'act' if 'act' in pdf_path.name.lower() else 'rule' if 'rule' in pdf_path.name.lower() else 'unknown'

    return {
        'text': full_text.strip(),
        'source': pdf_path.name,
        'document_type': doc_type,
        'tax_type': 'GST'
    }


def chunk_text(document: dict) -> list[dict]:
    '''Splits a large string into smaller chunks using LangChain.'''
    print(f'✂️ Chunking: {document["source"]}')
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=[r'(?i)\n(?=Section\s+\d+)', r'(?i)\n(?=Rule\s+\d+)', r'(?i)\n(?=CHAPTER)', '\n\n', '\n', '.', ' ', ''],
        is_separator_regex=True
    )
    
    chunks = text_splitter.split_text(document['text'])
    
    result = []
    for chunk in chunks:
        section_number = None
        section_title = None
        
        # Regex to detect "Section 73. Title" or "Rule 142 Title"
        match = re.search(r'^(?:Section|Rule)\s+(\d+[A-Z]*)[.\-\s]*([^\n]*)', chunk.strip(), flags=re.IGNORECASE)
        if match:
            section_number = match.group(1).strip()
            title = match.group(2).strip()
            if title:
                section_title = title

        meta = {
            'source': document['source'],
            'document_type': document.get('document_type', 'unknown'),
            'tax_type': document.get('tax_type', 'GST'),
            'section_number': section_number,
            'section_title': section_title
        }
        result.append({'text': chunk, 'metadata': meta})
        
    return result


def ingest_to_pgvector():
    '''Reads PDFs from data/, chunks them, embeds them, and saves to pgvector.'''
    print(f'🚀 Starting ingestion process...')

    import boto3
    import time
    
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )

    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id=BEDROCK_MODEL_ID
    )

    collection_name = 'gst_laws'
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    
    # 3. Process all PDFs in the data folder
    all_chunks = []
    
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f'❌ Data directory not found: {DATA_DIR}')
        return

    pdf_files = list(DATA_DIR.glob('*.pdf'))
    if not pdf_files:
        print(f'⚠️ No PDF files found in {DATA_DIR}')
        return

    print(f'📂 Found {len(pdf_files)} PDF files.')

    for pdf_path in pdf_files:
        doc = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
        print(f'✅ Generated {len(chunks)} chunks from {pdf_path.name}')

    if not all_chunks:
         print('⚠️ No content extracted to ingest.')
         return

    # 4. Ingest into database in small batches to avoid Bedrock rate limits.
    # Each batch calls the embedding API for every chunk, so large batches
    # cause silent throttling and make the script appear frozen.
    # We must use a much smaller batch size because Bedrock Titan has tight payload limits.
    # 50 chunks at once causes a massive payload that gets rate-limited/rejected indefinitely.
    SAFE_BATCH_SIZE = 5 
    print(f'💾 Upserting {total} total chunks in safe batches of {SAFE_BATCH_SIZE}...')

    texts = [chunk['text'] for chunk in all_chunks]
    metadatas = [chunk['metadata'] for chunk in all_chunks]

    for start in range(0, total, SAFE_BATCH_SIZE):
        end = min(start + SAFE_BATCH_SIZE, total)
        batch_texts = texts[start:end]
        batch_meta = metadatas[start:end]
        
        vector_store.add_texts(texts=batch_texts, metadatas=batch_meta)
        print(f'  ✅ Batch {start // SAFE_BATCH_SIZE + 1}/{(total + SAFE_BATCH_SIZE - 1) // SAFE_BATCH_SIZE} done ({end}/{total} chunks)')
        time.sleep(2) # Prevent Bedrock rate limits

    print('🎉 Ingestion complete!')

if __name__ == '__main__':
    ingest_to_pgvector()
