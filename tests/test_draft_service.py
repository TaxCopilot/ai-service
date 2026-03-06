import json
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from services.draft_service import generate_notice_reply

# Mock LangChain AIMessage response
class MockAIMessage:
    def __init__(self, content):
        self.content = content

@pytest.fixture
def mock_deps():
    with patch('services.draft_service.ChatBedrockConverse.invoke') as mock_invoke, \
         patch('boto3.Session') as mock_session, \
         patch('services.draft_service._extract_and_validate_citations', return_value=True) as mock_validate:
        yield mock_invoke, mock_validate

def test_json_extraction_clean(mock_deps):
    mock_invoke, _ = mock_deps
    mock_invoke.return_value = MockAIMessage('{"draft_reply": "Clean text.", "citations": ["Section 10"]}')
    
    response = generate_notice_reply(
        document_id="doc1",
        extracted_text="query",
        retrieved_law="law",
        unique_sources=["Section 73"]
    )
    
    assert response.draft_reply == "Clean text."
    # order is not guaranteed due to list(set(..)), but elements should match
    assert set(response.citations) == {"Section 10", "Section 73"}
    assert response.is_grounded is True

def test_json_extraction_with_structural_noise(mock_deps):
    mock_invoke, _ = mock_deps
    # LLM might hallucinates markdown codeblocks or newlines
    noisy_response = """
    Here is your draft:
    ```json
    {
        "draft_reply": "Text with newlines and noise.",
        "citations": ["Rule 3"]
    }
    ```
    """
    mock_invoke.return_value = MockAIMessage(noisy_response)
    
    response = generate_notice_reply(
        document_id="doc2",
        extracted_text="query",
        retrieved_law="law",
        unique_sources=["Section 73"]
    )
    
    assert response.draft_reply == "Text with newlines and noise."
    assert set(response.citations) == {"Rule 3", "Section 73"}

def test_json_extraction_malformed_fallback(mock_deps):
    mock_invoke, _ = mock_deps
    # Malformed JSON missing quotes, braces, etc.
    malformed_response = '{"draft_reply": "Malformed fallback text.", "citations": ["Section 80C"'
    mock_invoke.return_value = MockAIMessage(malformed_response)
    
    response = generate_notice_reply(
        document_id="doc3",
        extracted_text="query",
        retrieved_law="law",
        unique_sources=[]
    )
    
    # It should fall through the JSON decode exception, print a warning, and use regex fallback
    assert response.draft_reply == malformed_response
    assert set(response.citations) == {"Section 80C"}
    assert response.is_grounded is True
