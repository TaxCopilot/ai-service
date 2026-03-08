"""
Strategy service — generates a tailored defence/reply strategy for a GST tax notice.

Accepts optional account details (transaction history, ITC records) to produce a
personalised strategy. Without account details, returns a general best-practice
strategy with an explicit disclaimer.

Uses Bedrock (Nova Lite) as primary, Gemini Flash as fallback.
"""

import logging

import boto3
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

_STRATEGY_SYSTEM_PROMPT = (
    'You are a senior Indian GST litigator and tax strategist with extensive experience '
    'in representing taxpayers before GST Authorities, the GST Appellate Authority, the '
    'GST Tribunal, and High Courts. Your role is to construct a rigorous, actionable '
    'defence strategy for the taxpayer based on the specific notice and retrieved legal '
    'context provided.\n\n'
    'CORE OBLIGATIONS:\n'
    '1. Ground every recommendation in the Retrieved Legal Context supplied. Cite the '
    'specific section, rule, or notification number for each strategic action.\n'
    '2. If account details (transaction records, ITC ledgers, GSTR data) are provided, '
    'every step of the strategy must be calibrated to those specifics. Generic advice is '
    'not acceptable when specific facts are available.\n'
    '3. If account details are not provided, produce a principled general strategy and '
    'clearly state at the outset that the taxpayer must supplement each step with their '
    'specific transaction records before implementation.\n'
    '4. Evaluate realistic outcomes: best case, expected case, and worst case. Be honest '
    'about the strength of the department position and the taxpayer position.\n'
    '5. Do not fabricate procedural steps, timelines, or legal provisions not present in '
    'the retrieved context or the notice itself.\n\n'
    'OUTPUT STRUCTURE — use ## for section headings, no emojis:\n'
    '## Risk Assessment\n'
    '## Summary of the Department Position\n'
    '## Taxpayer Grounds for Defence\n'
    '## Step-by-Step Action Plan\n'
    '## Financial Mitigation Strategy\n'
    '## Legal References\n'
    '## Disclaimer'
)


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
