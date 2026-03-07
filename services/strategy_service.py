"""
Strategy service — generates a tailored defence/reply strategy for a GST tax notice.

Accepts optional account details (transaction history, ITC records) to produce a
personalised strategy. Without account details, returns a general best-practice
strategy with an explicit disclaimer.

Uses Bedrock (Nova Lite) as primary, Gemini Flash as fallback.
"""

import logging

from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

_GEMINI_MODEL = 'gemini-2.5-flash'
_RISK_KEYWORDS_HIGH = frozenset(
    ['high risk', 'high penalty', 'prosecution', 'fraud', 'evasion']
)
_RISK_KEYWORDS_LOW = frozenset(['low risk', 'minor discrepancy', 'clerical error'])

_STRATEGY_SYSTEM_PROMPT = """
You are a senior Indian GST lawyer and tax strategist. Create a clear, actionable
defence strategy for the taxpayer to tackle the given tax notice.

RULES:
1. Ground every recommendation in the retrieved legal context provided.
2. Cite the specific sections and rules that can be invoked in the taxpayer's favour.
3. If account details are provided, tailor every step to those specifics.
4. If account details are NOT provided, clearly state this is a GENERAL strategy and
   the taxpayer must consult their accountant for transaction-specific advice.
5. Be concrete — avoid generic advice.
6. Identify realistic ways to minimise the tax demand and penalty.

STRUCTURE your response with these exact headings (use ## for headings):
## Risk Assessment
## Strategy Overview
## Step-by-Step Action Plan
## Legal Grounds for Defence
## How to Minimise Loss
## Disclaimer
"""


class StrategyResponse(BaseModel):
    """Tailored defence strategy for a tax notice."""

    strategy: str = Field(description='Full strategy text in markdown format.')
    has_account_details: bool = Field(
        description='True if the strategy was personalised with account details.'
    )
    risk_level: str = Field(description='Assessed risk level: HIGH, MEDIUM, or LOW.')


def _invoke_with_fallback(messages: list) -> str:
    """Try Bedrock (Nova Lite) first; fall back to Gemini Flash on any failure."""
    try:
        logger.info('Strategy: attempting Bedrock LLM')
        llm = ChatBedrockConverse(
            model=settings.bedrock_model_id,
            region_name=settings.aws_region,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        return str(llm.invoke(messages).content)
    except (BotoCoreError, ClientError) as exc_bedrock:
        logger.warning(
            'Strategy: Bedrock failed — falling back to Gemini. Error: %s', exc_bedrock
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
                'Strategy: Both Bedrock and Gemini failed. Gemini error: %s', exc_gemini
            )
            raise RuntimeError(f'All LLM attempts failed: {exc_gemini}') from exc_gemini


def _detect_risk(content: str) -> str:
    """Detect risk level from the LLM output text."""
    lower = content.lower()
    if any(kw in lower for kw in _RISK_KEYWORDS_HIGH):
        return 'HIGH'
    if any(kw in lower for kw in _RISK_KEYWORDS_LOW):
        return 'LOW'
    return 'MEDIUM'


def _build_history_block(chat_history: list[dict]) -> str:
    """Serialize bounded chat history into a readable block for the LLM prompt."""
    if not chat_history:
        return ''
    lines = [f'{m["role"].upper()}: {m["content"]}' for m in chat_history]
    return 'Prior Conversation Context:\n\n' + '\n\n'.join(lines) + '\n\n'


def generate_strategy(
    document_id: str,
    extracted_text: str,
    retrieved_law: str,
    chat_history: list[dict],
    account_details: str | None = None,
) -> StrategyResponse:
    """
    Generate a tailored defence strategy for a GST tax notice.

    Uses retrieved law for legal grounding. Injects bounded prior chat history
    for context. Personalises the strategy if account_details are provided.
    """
    history_block = _build_history_block(chat_history)
    account_block = (
        f'Taxpayer Account Details:\n{account_details}\n\n'
        if account_details
        else (
            'Note: No account details provided. '
            'Generate a general best-practice strategy with a clear disclaimer.\n\n'
        )
    )
    prompt = (
        f'{history_block}'
        f'Retrieved Legal Context:\n\n{retrieved_law}\n\n'
        f'Tax Notice Text:\n\n{extracted_text}\n\n'
        f'{account_block}'
        f'Generate a comprehensive defence strategy for the above tax notice.'
    )
    messages = [('system', _STRATEGY_SYSTEM_PROMPT), ('user', prompt)]

    logger.info(
        'Strategy: generating for document_id=%s has_account_details=%s',
        document_id,
        bool(account_details),
    )
    content = _invoke_with_fallback(messages)

    return StrategyResponse(
        strategy=content,
        has_account_details=bool(account_details),
        risk_level=_detect_risk(content),
    )
