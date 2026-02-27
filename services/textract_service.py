"""
Extracts text from a notice PDF stored in S3 using AWS Textract.

Textract is used instead of a local PDF library because real Indian income
tax notices are often scanned — PyMuPDF returns garbled output on those.

Required IAM: textract:DetectDocumentText, s3:GetObject.
"""

import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import settings

logger = logging.getLogger(__name__)

_textract_client = None


def _get_client():
    global _textract_client
    if _textract_client is None:
        _textract_client = boto3.client("textract", region_name=settings.aws_region)
    return _textract_client


def extract_text_from_s3(s3_bucket: str, s3_key: str) -> str:
    """
    OCR a PDF from S3 and return its text content.

    Only LINE blocks are collected — including WORD blocks would duplicate
    every word in the output.

    Raises RuntimeError if Textract fails or the document has no text.
    """
    client = _get_client()
    logger.info("Textract | s3://%s/%s", s3_bucket, s3_key)

    try:
        response = client.detect_document_text(
            Document={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}}
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        logger.exception("Textract error [%s] for s3://%s/%s", code, s3_bucket, s3_key)
        raise RuntimeError(f"Textract failed ({code}) for s3://{s3_bucket}/{s3_key}") from exc
    except BotoCoreError as exc:
        logger.exception("Textract connection error")
        raise RuntimeError(f"Textract connection error: {exc}") from exc

    lines = [b["Text"] for b in response.get("Blocks", []) if b.get("BlockType") == "LINE"]

    if not lines:
        raise RuntimeError(
            f"Textract found no text in s3://{s3_bucket}/{s3_key}. "
            "The file may be blank, password-protected, or corrupt."
        )

    extracted = "\n".join(lines)
    logger.info("Textract extracted %d lines from s3://%s/%s", len(lines), s3_bucket, s3_key)
    return extracted
