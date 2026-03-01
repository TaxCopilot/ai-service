"""
Calls Amazon Bedrock to summarise a tax notice and draft a CA-grade reply.

The LLM is instructed to work only from the law excerpts provided — no
hallucinated citations. Output is expected as a raw JSON object.
"""

import json
import logging
import re

from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)

_LAW_FALLBACK = (
    'No specific law excerpts retrieved. '
    'Apply general knowledge of the Income Tax Act 1961.'
)
_JSON_FENCE_PATTERN = re.compile(r'```(?:json)?')
_PASSAGE_SEPARATOR = '\n\n---\n\n'
_TRUNCATION_NOTICE = '\n\n[Notice text truncated to stay within the model context limit.]'

_SYSTEM_PROMPT = """\
You are an Indian Chartered Accountant with 20 years of direct tax litigation experience.

Analyse the income tax notice provided and produce a structured response.

Rules:
- Work strictly from the provided law excerpts. Do not cite provisions not present in them.
- Use formal language appropriate for correspondence with the Income Tax Department.
- Reply with a raw JSON object containing exactly two keys:
  {
    "generated_summary": "<plain-English explanation: what the notice demands, why it was issued, deadlines>",
    "draft_response": "<complete formal reply letter to the Assessing Officer>"
  }

No markdown fences. No preamble. JSON only.\
"""

_USER_PROMPT = """\
--- NOTICE ---
{notice_text}

--- RELEVANT LAW ---
{law_excerpts}

Produce the JSON response.\
"""


class NoticeResponse(BaseModel):
    document_id: str
    generated_summary: str
    draft_response: str
    sources_cited: list[str]


def _build_llm() -> ChatBedrock:
    import boto3
    import botocore.config

    boto_config = botocore.config.Config(
        region_name=settings.aws_region,
        retries={'max_attempts': 3, 'mode': 'standard'}
    )
    
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=boto_config,
    )

    return ChatBedrock(
        client=bedrock_client,
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
        model_kwargs={
            'max_tokens': settings.bedrock_max_tokens,
            'temperature': settings.bedrock_temperature,
        },
    )


def _parse_json(raw: str) -> dict[str, str]:
    """Extract the JSON object from the LLM response, stripping any markdown fences."""
    cleaned = _JSON_FENCE_PATTERN.sub('', raw).strip().rstrip('`').strip()
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM response: {raw[:300]!r}")
    try:
        return json.loads(match.group())  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise ValueError(f'Malformed JSON in LLM response: {exc}') from exc


def _truncate_notice(text: str, max_chars: int) -> str:
    """Cap notice text to max_chars to avoid exceeding the model's context window."""
    if len(text) <= max_chars:
        return text
    logger.warning(
        'Notice text truncated from %d to %d characters to fit context window.',
        len(text), max_chars,
    )
    return text[:max_chars] + _TRUNCATION_NOTICE


def generate_notice_reply(
    document_id: str,
    extracted_text: str,
    retrieved_law: str,
    sources_cited: list[str],
) -> NoticeResponse:
    """
    Invoke Bedrock and parse the structured reply.

    Raises RuntimeError if the model call fails or the response cannot be parsed.
    """
    llm = _build_llm()
    extracted_text = _truncate_notice(extracted_text, settings.bedrock_max_notice_chars)
    law_context = retrieved_law.strip() or _LAW_FALLBACK

    messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=_USER_PROMPT.format(
                notice_text=extracted_text,
                law_excerpts=law_context,
            )
        ),
    ]

    logger.info('Invoking Bedrock | model=%s', settings.bedrock_model_id)

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.exception('Bedrock invocation failed')
        raise RuntimeError(f'Bedrock error: {exc}') from exc

    raw: str = response.content  # type: ignore[assignment]
    logger.debug('Bedrock response (first 500): %.500s', raw)

    try:
        parsed = _parse_json(raw)
    except ValueError as exc:
        logger.exception('Failed to parse Bedrock response')
        raise RuntimeError(str(exc)) from exc

    return NoticeResponse(
        document_id=document_id,
        generated_summary=parsed.get('generated_summary', ''),
        draft_response=parsed.get('draft_response', ''),
        sources_cited=sources_cited,
    )
