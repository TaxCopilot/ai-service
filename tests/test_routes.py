import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from config import settings

client = TestClient(app)

# Dummy payloads
VALID_DECODE_REQUEST = {
    "mode": "decode",
    "document_id": "test_doc_123",
    "s3_bucket": "test-bucket",
    "s3_key": "test/path.pdf"
}

VALID_CHAT_REQUEST = {
    "mode": "chat",
    "message": "What is the penalty for late filing?"
}

HEADERS = {"X-API-Key": settings.api_key} if settings.api_key else {}

@patch('api.routes.get_cached_doc')
def test_db_failure_cache_lookup(mock_get_cached_doc):
    # Simulate DB failure during cache lookup -> 503 with stage "cache"
    mock_get_cached_doc.side_effect = RuntimeError("DB connection dropped")
    
    response = client.post("/api/v1/ask", json=VALID_DECODE_REQUEST, headers=HEADERS)
    
    assert response.status_code == 503
    data = response.json()
    assert getattr(data, "get", lambda x: {})( "detail", {}).get("stage") == "cache"
    assert "DB connection dropped" in data["detail"]["error"]

@patch('api.routes.get_cached_doc', return_value=None)
@patch('api.routes.extract_text_from_s3')
def test_textract_failure(mock_extract, mock_get_cached_doc):
    # Simulate Textract failure -> 422 with stage "textract"
    mock_extract.side_effect = RuntimeError("Textract job failed")
    
    response = client.post("/api/v1/ask", json=VALID_DECODE_REQUEST, headers=HEADERS)
    
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["stage"] == "textract"
    assert "Textract job failed" in data["detail"]["error"]

@patch('api.routes.retrieve_relevant_law')
def test_kb_failure(mock_retrieve):
    # Simulate KB failure (e.g. Bedrock error during embedding, or PGVector error) -> 503 with stage "knowledge_base"
    mock_retrieve.side_effect = RuntimeError("Vector DB unreachable")
    
    response = client.post("/api/v1/ask", json=VALID_CHAT_REQUEST, headers=HEADERS)
    
    assert response.status_code == 503
    data = response.json()
    assert data["detail"]["stage"] == "knowledge_base"
    assert "Vector DB unreachable" in data["detail"]["error"]

@patch('api.routes.get_cached_doc', return_value=None)
@patch('api.routes.extract_text_from_s3', return_value="Dummy text")
@patch('api.routes.retrieve_relevant_law', return_value=(["law text"], ["source1"]))
@patch('api.routes.generate_notice_reply')
def test_llm_failure(mock_generate, mock_retrieve, mock_extract, mock_cache):
    # Simulate LLM failure during generation -> 502 with stage "llm_generation"
    mock_generate.side_effect = RuntimeError("Bedrock ThrottlingException")
    
    response = client.post("/api/v1/ask", json=VALID_DECODE_REQUEST, headers=HEADERS)
    
    assert response.status_code == 502
    data = response.json()
    assert data["detail"]["stage"] == "llm_generation"
    assert "Bedrock ThrottlingException" in data["detail"]["error"]
