"""
Extracts text from a notice PDF stored in S3.

Primary: AWS Textract (best for scanned documents)
Fallback: PyPDF2 local extraction (for digital PDFs when Textract unavailable)

Required IAM: textract:DetectDocumentText, s3:GetObject.
"""

import io
import logging
import tempfile

import boto3
import botocore.client
from botocore.exceptions import BotoCoreError, ClientError

from config import settings

logger = logging.getLogger(__name__)

_TEXTRACT_SERVICE = 'textract'
_S3_SERVICE = 's3'
_BLOCK_TYPE_LINE = 'LINE'

_textract_client: botocore.client.BaseClient | None = None
_s3_client: botocore.client.BaseClient | None = None


def _get_boto_credentials() -> dict:
    """Get AWS credentials from settings for boto3 clients."""
    creds = {'region_name': settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        creds['aws_access_key_id'] = settings.aws_access_key_id
        creds['aws_secret_access_key'] = settings.aws_secret_access_key
    return creds


def _get_textract_client() -> botocore.client.BaseClient:
    global _textract_client
    if _textract_client is None:
        _textract_client = boto3.client(_TEXTRACT_SERVICE, **_get_boto_credentials())
    return _textract_client


def _get_s3_client() -> botocore.client.BaseClient:
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(_S3_SERVICE, **_get_boto_credentials())
    return _s3_client


def _extract_with_pymupdf(s3_bucket: str, s3_key: str) -> str:
    """Fallback: Download from S3 and extract text using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError('PyMuPDF not installed. Run: pip install pymupdf')
    
    logger.info('Using PyMuPDF fallback for s3://%s/%s', s3_bucket, s3_key)
    
    s3 = _get_s3_client()
    response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    pdf_bytes = response['Body'].read()
    
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    text_parts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
    doc.close()
    
    if not text_parts:
        raise RuntimeError(f'No text extracted from s3://{s3_bucket}/{s3_key}')
    
    extracted = '\n'.join(text_parts)
    logger.info('PyMuPDF extracted %d characters from s3://%s/%s', len(extracted), s3_bucket, s3_key)
    return extracted


def extract_text_from_s3(s3_bucket: str, s3_key: str) -> str:
    """
    Extract text from a PDF in S3.
    
    Attempts Textract first (best for scanned docs), falls back to PyPDF2.
    Raises RuntimeError if all methods fail.
    """
    client = _get_textract_client()
    logger.info('Textract | s3://%s/%s', s3_bucket, s3_key)

    try:
        response = client.detect_document_text(
            Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}}
        )
    except ClientError as exc:
        code = exc.response['Error']['Code']
        logger.warning('Textract error [%s] for s3://%s/%s - trying PyMuPDF fallback', code, s3_bucket, s3_key)
        # Fallback to PyMuPDF for credential/permission/format issues
        fallback_codes = (
            'UnrecognizedClientException',
            'AccessDeniedException', 
            'InvalidAccessKeyId',
            'UnsupportedDocumentException',  # PDF format not supported by Textract
            'BadDocumentException',          # Corrupted document
            'DocumentTooLargeException',     # Document too large
        )
        if code in fallback_codes:
            return _extract_with_pymupdf(s3_bucket, s3_key)
        raise RuntimeError(f'Textract failed ({code}) for s3://{s3_bucket}/{s3_key}') from exc
    except BotoCoreError as exc:
        logger.warning('Textract connection error - trying PyMuPDF fallback')
        return _extract_with_pymupdf(s3_bucket, s3_key)

    all_lines = [
        b
        for b in response.get('Blocks', [])
        if b.get('BlockType') == _BLOCK_TYPE_LINE
    ]

    if not all_lines:
        logger.warning('Textract found no text - trying PyMuPDF fallback')
        return _extract_with_pymupdf(s3_bucket, s3_key)

    threshold: float = settings.textract_min_confidence
    high_conf_lines = [b['Text'] for b in all_lines if b.get('Confidence', 0.0) >= threshold]

    if high_conf_lines:
        lines = high_conf_lines
        dropped = len(all_lines) - len(lines)
        if dropped:
            logger.debug(
                'Dropped %d low-confidence lines (threshold=%.0f) from s3://%s/%s',
                dropped, threshold, s3_bucket, s3_key,
            )
    else:
        logger.warning(
            'All %d lines are below confidence threshold %.0f for s3://%s/%s — '
            'using full Textract output.',
            len(all_lines), threshold, s3_bucket, s3_key,
        )
        lines = [b['Text'] for b in all_lines]
        dropped = 0

    extracted = '\n'.join(lines)
    logger.info(
        'Textract extracted %d lines (dropped %d low-confidence) from s3://%s/%s',
        len(lines), dropped, s3_bucket, s3_key,
    )
    return extracted
