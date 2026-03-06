import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from services.textract_service import extract_text_from_s3
from config import settings

@pytest.fixture
def mock_textract():
    with patch('services.textract_service._get_client') as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock start document text detection
        mock_client.start_document_text_detection.return_value = {'JobId': 'test-job'}
        yield mock_client

def test_line_filtering_drops_words_and_low_confidence(mock_textract):
    # Prepare blocks: 2 High-conf LINEs, 1 Low-conf LINE, 1 High-conf WORD
    mock_textract.get_document_text_detection.return_value = {
        'JobStatus': 'SUCCEEDED',
        'Blocks': [
            {'BlockType': 'LINE', 'Text': 'High Conf Line 1', 'Confidence': 99.0},
            {'BlockType': 'LINE', 'Text': 'Low Conf Line', 'Confidence': settings.textract_min_confidence - 10.0},
            {'BlockType': 'LINE', 'Text': 'High Conf Line 2', 'Confidence': 95.0},
            {'BlockType': 'WORD', 'Text': 'DiscardedWord', 'Confidence': 99.0},
        ],
        'NextToken': None
    }
    
    extracted_text = extract_text_from_s3('test-bucket', 'test-key')
    
    # Assert 'Low Conf Line' and 'DiscardedWord' are gone
    expected = "High Conf Line 1\nHigh Conf Line 2"
    assert extracted_text == expected

def test_all_low_confidence_fallback(mock_textract):
    # Prepare blocks: All LINEs are low confidence
    mock_textract.get_document_text_detection.return_value = {
        'JobStatus': 'SUCCEEDED',
        'Blocks': [
            {'BlockType': 'LINE', 'Text': 'Garbage 1', 'Confidence': 10.0},
            {'BlockType': 'LINE', 'Text': 'Garbage 2', 'Confidence': 12.0},
        ],
        'NextToken': None
    }
    
    extracted_text = extract_text_from_s3('test-bucket', 'test-key')
    
    # Assert fallback kicks in and BOTH lines are kept
    expected = "Garbage 1\nGarbage 2"
    assert extracted_text == expected

def test_empty_document_raises_runtime_error(mock_textract):
    # No blocks returned
    mock_textract.get_document_text_detection.return_value = {
        'JobStatus': 'SUCCEEDED',
        'Blocks': [],
        'NextToken': None
    }
    
    with pytest.raises(RuntimeError, match="Textract found no text in s3://test-bucket/test-key"):
        extract_text_from_s3('test-bucket', 'test-key')
