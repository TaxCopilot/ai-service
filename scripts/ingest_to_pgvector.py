import os
from pathlib import Path
from dotenv import load_dotenv

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_postgres.vectorstores import PGVector

# Load environment variables explicitly
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"

# LangChain's PGVector needs psycopg plugin for psycopg3
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")

# Explicitly use the Titan Embedding v2 model. 
# Do not default to the Claude LLM model id defined in .env
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from environment variables")


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """Extracts text from a PDF file using PyMuPDF and returns a dict with metadata."""
    print(f"📄 Processing: {pdf_path.name}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += text + "\n"
    
    return {
        "text": full_text,
        "source": pdf_path.name
    }


def chunk_text(document: dict) -> list[dict]:
    """Splits a large string into smaller chunks using LangChain."""
    print(f"✂️ Chunking: {document['source']}")
    
    # Tax laws are dense; 1000 characters with 100 overlap is a good balance
    # for capturing context without losing specificity.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(document["text"])
    
    # Re-attach metadata to each chunk
    return [{"text": chunk, "source": document["source"]} for chunk in chunks]


def ingest_to_pgvector():
    """Reads PDFs from data/, chunks them, embeds them, and saves to pgvector."""
    print(f"🚀 Starting ingestion process...")

    # 1. Setup Embeddings
    print(f"🤖 Initializing Bedrock Embeddings ({BEDROCK_MODEL_ID})...")
    embeddings = BedrockEmbeddings(model_id=BEDROCK_MODEL_ID)

    # 2. Setup Vector Store Connection
    print("🔌 Connecting to PostgreSQL...")
    collection_name = "tax_laws"
    
    # Create the vector store. It will automatically create tables if they don't exist.
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    
    # 3. Process all PDFs in the data folder
    all_chunks = []
    
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDF files found in {DATA_DIR}")
        return

    print(f"📂 Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        # Extract
        doc = extract_text_from_pdf(pdf_path)
        
        # Chunk
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
        print(f"✅ Generated {len(chunks)} chunks from {pdf_path.name}")

    if not all_chunks:
         print("⚠️ No content extracted to ingest.")
         return

    # 4. Ingest into database in small batches to avoid Bedrock rate limits.
    # Each batch calls the embedding API for every chunk, so large batches
    # cause silent throttling and make the script appear frozen.
    BATCH_SIZE = 50
    total = len(all_chunks)
    print(f'💾 Upserting {total} total chunks in batches of {BATCH_SIZE}...')

    texts = [chunk['text'] for chunk in all_chunks]
    metadatas = [{'source': chunk['source']} for chunk in all_chunks]

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_texts = texts[start:end]
        batch_meta = metadatas[start:end]
        vector_store.add_texts(texts=batch_texts, metadatas=batch_meta)
        print(f'  ✅ Batch {start // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE} done ({end}/{total} chunks)')

    print('🎉 Ingestion complete!')

if __name__ == "__main__":
    ingest_to_pgvector()
