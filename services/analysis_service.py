"""
Analysis service — performs a deep structured analysis on a GST tax notice.

Returns a rich breakdown: notice type, sections applied, demands, deadlines,
and recommended immediate actions. Uses Bedrock (Nova Lite) as primary, with
Gemini Flash as fallback.
"""

import json
import logging
import re

from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

_GEMINI_MODEL = 'gemini-2.5-flash'

_ANALYSIS_SYSTEM_PROMPT = """
You are a senior Indian GST and Income Tax legal analyst.
Your task is to perform a deep, structured analysis of a tax notice.

RULES:
1. Only use information present in the notice text and the retrieved legal context.
2. Extract exact section numbers, rule numbers, and legal references from the notice.
3. Be specific about deadlines, demand amounts, and penalties mentioned.
4. Do NOT hallucinate any figures or sections not present in the text.

OUTPUT FORMAT — respond ONLY with this exact JSON (no extra text, no markdown fences):
{
  "notice_type": "e.g. ASMT-10, Section 73 SCN, etc.",
  "summary": "2-3 sentence plain-language explanation of what this notice means for the taxpayer",
  "sections_applied": ["Section 73(1) CGST Act", "Rule 142"],
  "demands": [
    {"description": "Tax demand for FY 2022-23", "amount": "Rs. 2,45,000"}
  ],
  "deadline": "30 days from date of notice / specific date if mentioned",
  "immediate_actions": [
    "File a reply within 30 days",
    "Gather ITC records for FY 2022-23"
  ],
  "risk_level": "HIGH",
  "citations": ["Section 73", "Rule 142"]
}
"""


class DemandItem(BaseModel):
    """A single monetary demand from the notice."""

    description: str
    amount: str


class AnalysisResponse(BaseModel):
    """Structured deep analysis of a GST / Income Tax notice."""

    notice_type: str = Field(
        description='Type of notice, e.g. ASMT-10, Section 73 SCN.'
    )
    summary: str = Field(description='Plain-language summary of what the notice means.')
    sections_applied: list[str] = Field(
        description='Legal sections and rules cited in the notice.'
    )
    demands: list[DemandItem] = Field(description='Monetary demands with descriptions.')
    deadline: str = Field(description='Reply or compliance deadline.')
    immediate_actions: list[str] = Field(
        description='Recommended immediate actions for the taxpayer.'
    )
    risk_level: str = Field(description='Overall risk: HIGH, MEDIUM, or LOW.')
    citations: list[str] = Field(description='Legal sources grounding this analysis.')


def _invoke_with_fallback(messages: list) -> str:
    """Try Bedrock (Nova Lite) first; fall back to Gemini Flash on any failure."""
    try:
        logger.info('Analysis: attempting Bedrock LLM')
        llm = ChatBedrockConverse(
            model=settings.bedrock_model_id,
            region_name=settings.aws_region,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        return str(llm.invoke(messages).content)
    except (BotoCoreError, ClientError) as exc_bedrock:
        logger.warning(
            'Analysis: Bedrock failed — falling back to Gemini. Error: %s', exc_bedrock
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
                'Analysis: Both Bedrock and Gemini failed. Gemini error: %s', exc_gemini
            )
            raise RuntimeError(f'All LLM attempts failed: {exc_gemini}') from exc_gemini


def generate_analysis(
    document_id: str,
    extracted_text: str,
    retrieved_law: str,
    unique_sources: list[str],
) -> AnalysisResponse:
    """
    Perform a deep structured analysis of a tax notice.

    Returns a rich breakdown including notice type, sections applied,
    demands, deadlines, risk level, and recommended immediate actions.
    """
    prompt = (
        f'Retrieved Legal Context:\n\n{retrieved_law}\n\n'
        f'Tax Notice Text to Analyse:\n\n{extracted_text}'
    )
    messages = [('system', _ANALYSIS_SYSTEM_PROMPT), ('user', prompt)]

    logger.info('Analysis: generating for document_id=%s', document_id)
    content = _invoke_with_fallback(messages)

    # Strip markdown code fences the LLM may add despite instructions
    content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
    content = re.sub(r'\s*```$', '', content.strip())

    try:
        data = json.loads(content)
        demands = [
            DemandItem(
                description=d.get('description', ''),
                amount=d.get('amount', ''),
            )
            for d in data.get('demands', [])
        ]
        return AnalysisResponse(
            notice_type=data.get('notice_type', 'Unknown'),
            summary=data.get('summary', ''),
            sections_applied=data.get('sections_applied', []),
            demands=demands,
            deadline=data.get('deadline', 'Not specified'),
            immediate_actions=data.get('immediate_actions', []),
            risk_level=data.get('risk_level', 'MEDIUM'),
            citations=list(set(data.get('citations', []) + unique_sources)),
        )
    except json.JSONDecodeError as exc:
        logger.warning(
            'Analysis: LLM output was not valid JSON for %s: %s', document_id, exc
        )
        # Graceful fallback — surface raw text as the summary
        return AnalysisResponse(
            notice_type='Unknown',
            summary=content,
            sections_applied=[],
            demands=[],
            deadline='Not specified',
            immediate_actions=[],
            risk_level='MEDIUM',
            citations=unique_sources,
        )
