"""
Extracts text from a notice PDF stored in S3 using AWS Textract.

Textract is used instead of a local PDF library because real Indian income
tax notices are often scanned — PyMuPDF returns garbled output on those.

Required IAM: textract:DetectDocumentText, s3:GetObject.
"""

import logging

import boto3
import botocore.client
from botocore.exceptions import BotoCoreError, ClientError

from config import settings

logger = logging.getLogger(__name__)

_TEXTRACT_SERVICE = 'textract'
_BLOCK_TYPE_LINE = 'LINE'
_LINE_SEPARATOR = '\n'

_textract_client: botocore.client.BaseClient | None = None


def _get_client() -> botocore.client.BaseClient:
    global _textract_client
    if _textract_client is None:
        _textract_client = boto3.client(_TEXTRACT_SERVICE, region_name=settings.aws_region)
    return _textract_client


def extract_text_from_s3(s3_bucket: str, s3_key: str) -> str:
    """
    OCR a PDF from S3 and return its text content.

    Only LINE blocks are collected — including WORD blocks would duplicate
    every word in the output.

    Raises RuntimeError if Textract fails or the document has no text.
    """
    client = _get_client()
    logger.info('Textract | s3://%s/%s', s3_bucket, s3_key)

    try:
        response = client.detect_document_text(
            Document={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}}
        )
    except ClientError as exc:
        code = exc.response['Error']['Code']
        logger.exception('Textract error [%s] for s3://%s/%s', code, s3_bucket, s3_key)
        raise RuntimeError(f'Textract failed ({code}) for s3://{s3_bucket}/{s3_key}') from exc
    except BotoCoreError as exc:
        logger.exception('Textract connection error')
        raise RuntimeError(f'Textract connection error: {exc}') from exc

    all_lines = [
        b
        for b in response.get('Blocks', [])
        if b.get('BlockType') == _BLOCK_TYPE_LINE
    ]

    if not all_lines:
        raise RuntimeError(
            f'Textract found no text in s3://{s3_bucket}/{s3_key}. '
            'The file may be blank, password-protected, or corrupt.'
        )

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
        # Every line is below the threshold — likely a very poor scan (old fax, low ink).
        # Use all output rather than returning nothing; the LLM handles noisy text better
        # than a hard failure.
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
