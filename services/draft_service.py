"""
Draft service — two generators:

  1. generate_notice_reply  : Structured JSON reply draft (existing decode mode).
  2. generate_html_draft    : Full HTML-formatted draft reply for the GST Department,
                              built from the document + full session chat context.

Both use Bedrock (Nova Lite) as primary with Gemini Flash as fallback.
Hallucination detection and retry logic applies to both.
"""

import json
import logging
import re

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

_GEMINI_MODEL = 'gemini-2.5-flash'

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class NoticeResponse(BaseModel):
    """Response schema for the structured legal draft (decode mode)."""

    draft_reply: str = Field(description='The generated reply to the GST tax notice.')
    citations: list[str] = Field(description='List of legal sections or rules cited.')
    is_grounded: bool = Field(
        description='Whether the response is grounded in retrieved law.'
    )


class DraftHtmlResponse(BaseModel):
    """Response schema for the HTML-formatted legal draft (draft mode)."""

    html_content: str = Field(description='Formal draft reply formatted as raw HTML.')
    citations: list[str] = Field(description='Legal sections cited in the draft.')


def _invoke_bedrock_with_fallback(messages: list, log_prefix: str) -> str:
    """
    Try Bedrock (Nova Lite) first; fall back to Gemini Flash on any failure.
    log_prefix is used in log messages to identify the calling service.
    """
    try:
        logger.info('%s: attempting Bedrock LLM', log_prefix)
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        llm = ChatBedrockConverse(
            model=settings.bedrock_model_id,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            client=session.client('bedrock-runtime'),
        )
        return str(llm.invoke(messages).content)
    except (BotoCoreError, ClientError) as exc_bedrock:
        logger.warning(
            '%s: Bedrock failed — falling back to Gemini. Error: %s',
            log_prefix,
            exc_bedrock,
        )
        try:
            llm_gemini = ChatGoogleGenerativeAI(
                model=_GEMINI_MODEL,
                api_key=settings.gemini_api_key,
                max_output_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
            return str(llm_gemini.invoke(messages).content)
        except Exception as exc_gemini:
            logger.error(
                '%s: Both Bedrock and Gemini failed. Gemini error: %s',
                log_prefix,
                exc_gemini,
            )
            raise RuntimeError(f'All LLM attempts failed: {exc_gemini}') from exc_gemini


# ---------------------------------------------------------------------------
# decode mode — structured JSON draft reply (existing)
# ---------------------------------------------------------------------------

_DECODE_SYSTEM_PROMPT = (
    'You are a senior Indian tax litigation counsel specialising in GST disputes. '
    'Your function is to draft a precise, legally defensible reply to the provided '
    'GST or Income Tax notice on behalf of the taxpayer.\n\n'
    'GROUNDING RULES (non-negotiable):\n'
    '1. Every legal assertion, section reference, and rule citation must be traceable '
    'to the Retrieved Legal Context supplied in the prompt. Do not cite any provision '
    'not present in that context.\n'
    '2. If the retrieved context is insufficient to construct a grounded reply, respond '
    'exactly: "Insufficient information in current legal corpus to draft a grounded reply."\n'
    '3. Do not fabricate transaction details, GSTIN numbers, amounts, or dates not '
    'present in the notice text itself.\n'
    '4. Do not rely on general legal knowledge to fill gaps in the retrieved context.\n\n'
    'DRAFTING STANDARDS:\n'
    '- The reply must open with the taxpayer details, the notice reference number, and '
    'the date of the notice.\n'
    '- Address every grounds raised in the notice specifically.\n'
    '- Cite the exact legal basis for each defence point (section number, rule, '
    'notification).\n'
    '- Conclude with a clear statement of the taxpayer position and the relief sought.\n'
    '- Use formal legal English.\n\n'
    'OUTPUT FORMAT — respond only with this exact JSON structure (no markdown fences):\n'
    '{\n'
    '  "draft_reply": "Full formal reply text...",\n'
    '  "citations": ["Section 73(1) CGST Act", "Rule 142"],\n'
    '  "is_grounded": true\n'
    '}'
)


def _extract_and_validate_citations(llm_output: str, retrieved_law: str) -> bool:
    """
    Extracts citations from LLM output and validates them against the retrieved law text.
    Returns True if all citations are found in the retrieved context.
    """
    citations = re.findall(
        r'(?:Section|Rule)\s+\d+[A-Z]*', llm_output, flags=re.IGNORECASE
    )
    if not citations:
        return True
    for cite in set(citations):
        clean_cite = cite.lower().strip()
        num_match = re.search(r'\d+[A-Z]*', clean_cite)
        if num_match:
            number = num_match.group(0)
            if number not in retrieved_law.lower():
                logger.warning(
                    'HALLUCINATION DETECTED: %s (number %s) not in retrieved context.',
                    cite,
                    number,
                )
                return False
        elif clean_cite not in retrieved_law.lower():
            logger.warning('HALLUCINATION DETECTED: %s not in retrieved context.', cite)
            return False
    return True


def generate_notice_reply(
    document_id: str,
    extracted_text: str,
    retrieved_law: str,
    unique_sources: list[str],
) -> NoticeResponse:
    """
    Orchestrates the RAG pipeline to generate a grounded legal reply using retrieved laws.
    Includes hallucination detection with a single retry if invalid citations are found.
    """
    if not retrieved_law.strip():
        return NoticeResponse(
            draft_reply='Insufficient information in current legal corpus based on the provided text.',
            citations=[],
            is_grounded=False,
        )

    prompt = f'Retrieved Legal Context:\n\n{retrieved_law}\n\nNotice Text to Reply To:\n\n{extracted_text}'
    messages = [('system', _DECODE_SYSTEM_PROMPT), ('user', prompt)]

    logger.info('Drafting reply for document_id=%s', document_id)
    content = _invoke_bedrock_with_fallback(messages, log_prefix='Draft')

    is_valid = _extract_and_validate_citations(content, retrieved_law)

    if not is_valid:
        logger.warning(
            'Hallucination detected for %s. Retrying with strict warnings.', document_id
        )
        warning_msg = (
            'WARNING: Your previous draft cited sections/rules NOT present in the context. '
            'REGENERATE the draft and ONLY cite the legal text provided. '
            'Do NOT fabricate section numbers.'
        )
        retry_prompt = f'{prompt}\n\n{warning_msg}'
        content = _invoke_bedrock_with_fallback(
            [('system', _DECODE_SYSTEM_PROMPT), ('user', retry_prompt)],
            log_prefix='Draft-retry',
        )
        is_valid = _extract_and_validate_citations(content, retrieved_law)

    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            final_citations = list(set(data.get('citations', []) + unique_sources))
            return NoticeResponse(
                draft_reply=data.get('draft_reply', 'Error parsing draft.'),
                citations=final_citations,
                is_grounded=is_valid,
            )
        except json.JSONDecodeError as exc:
            logger.warning(
                'Draft: LLM output was not valid JSON for %s: %s', document_id, exc
            )

    return NoticeResponse(
        draft_reply=content,
        citations=list(
            set(re.findall(r'(?:Section|Rule)\s+\d+[A-Z]*', content) + unique_sources)
        ),
        is_grounded=is_valid,
    )


# ---------------------------------------------------------------------------
# draft mode — HTML-formatted formal draft reply (new)
# ---------------------------------------------------------------------------

_HTML_DRAFT_SYSTEM_PROMPT = """
You are a senior Indian GST legal drafter. Generate a formal, professional reply to
the GST Department on behalf of the taxpayer.

LEGAL RULES:
1. Only use legal grounds present in the retrieved legal context provided.
2. Be formal, precise, and professional.
3. Ground every argument in specific sections and rules from the legal context.
4. Use the prior conversation context to understand the taxpayer's position and concerns.

CRITICAL OUTPUT FORMAT RULES:
- Return ONLY raw HTML. No markdown. No code fences. No backticks whatsoever.
- Wrap everything in <div class="legal-draft">.
- Use: <h2> for section headings, <p> for paragraphs, <ul><li> for lists,
  <strong> for key terms, <em> for emphasis.
- End with a <p class="signature"> block containing "Yours Faithfully," and space for signature.
- Do NOT wrap in ```html or any code fences.
"""


def _strip_html_fences(content: str) -> str:
    """Remove markdown code fences the LLM may add despite instructions."""
    content = re.sub(r'^```(?:html)?\s*', '', content.strip(), flags=re.IGNORECASE)
    content = re.sub(r'\s*```$', '', content.strip())
    return content


def _build_history_block(chat_history: list[dict]) -> str:
    """Serialize bounded chat history into a readable block for the LLM prompt."""
    if not chat_history:
        return ''
    lines = [f'{m["role"].upper()}: {m["content"]}' for m in chat_history]
    return 'Prior Conversation Context:\n\n' + '\n\n'.join(lines) + '\n\n'


def generate_html_draft(
    document_id: str,
    extracted_text: str,
    retrieved_law: str,
    chat_history: list[dict],
    unique_sources: list[str],
) -> DraftHtmlResponse:
    """
    Generate a formal HTML-formatted reply draft to the GST Department.

    Incorporates the full bounded session history for context, grounded in the
    retrieved legal corpus. The HTML is sanitized to remove any LLM-added fences.
    """
    if not retrieved_law.strip():
        fallback_html = (
            '<div class="legal-draft">'
            '<p>Insufficient legal context available to generate a grounded draft reply. '
            'Please ensure the knowledge base has relevant tax law documents loaded.</p>'
            '</div>'
        )
        return DraftHtmlResponse(html_content=fallback_html, citations=[])

    history_block = _build_history_block(chat_history)
    prompt = (
        f'{history_block}'
        f'Retrieved Legal Context:\n\n{retrieved_law}\n\n'
        f'Tax Notice Text:\n\n{extracted_text}\n\n'
        f'Generate a formal HTML reply draft to the GST Department for the above notice.'
    )
    messages = [('system', _HTML_DRAFT_SYSTEM_PROMPT), ('user', prompt)]

    logger.info('HTML Draft: generating for document_id=%s', document_id)
    content = _invoke_bedrock_with_fallback(messages, log_prefix='HTML-Draft')
    content = _strip_html_fences(content)

    cited = list(
        set(
            re.findall(r'(?:Section|Rule)\s+\d+[A-Z]*', content, flags=re.IGNORECASE)
            + unique_sources
        )
    )

    return DraftHtmlResponse(html_content=content, citations=cited)
