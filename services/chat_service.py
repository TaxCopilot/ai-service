'''
Chat service for answering general tax-law questions using RAG + Gemini.

This is the default 'chat' mode — the user asks a freeform question and
Gemini answers using passages retrieved from the tax_laws vector store.
No document upload or OCR required.
'''

import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    '''Response schema for a chat-mode answer.'''

    answer: str = Field(
        description='Plain-language answer to the user question, grounded in tax law.',
    )
    citations: list[str] = Field(
        description='Legal sources cited in the answer.',
    )

_CHAT_SYSTEM_PROMPT = (
    'You are TaxCopilot, a specialist Indian tax law assistant with deep expertise in '
    'the CGST Act, SGST Act, IGST Act, GST Rules, Income Tax Act 1961, and related '
    'notifications and circulars. You serve tax professionals, lawyers, and businesses '
    'navigating Indian tax notices and disputes.\n\n'
    'CONTEXT HIERARCHY (in order of authority):\n'
    '1. Retrieved Legal Context — specific provisions from the legal corpus. These are '
    'your primary evidentiary source. Cite sections directly.\n'
    '2. Active Document Context — the tax notice or document under review. Use this to '
    'answer document-specific questions.\n'
    '3. Recent Chat History — prior exchanges in this session. Use for continuity.\n\n'
    'OPERATING RULES:\n'
    '1. Base every legal assertion on the provided context. Do not fabricate section '
    'numbers, rule references, notification numbers, or monetary figures.\n'
    '2. If a legal question cannot be answered from the provided context, say so '
    'explicitly: "The retrieved context does not contain sufficient information to '
    'answer this with certainty."\n'
    '3. If the user requests drafting, summarisation, or analysis of the active document, '
    'you must fulfil the request using the document context provided.\n'
    '4. CONTEXTUAL ANCHORING: Even when the user asks a general question (e.g., "what do I need '
    'to provide for a draft?"), you MUST anchor your answer in the facts of the Active Document '
    'Context. Explicitly mention the names of the parties involved (e.g., K K FABRICS), specific '
    'monetary figures, mismatched invoices, or specific allegations mentioned in the document.\n'
    '5. Cite the specific provision (section number, rule, sub-clause) at the end of '
    'any legal answer.\n'
    '6. Be direct and precise. Do not hedge with unnecessary caveats when the law is clear.\n\n'
    'UI STRUCTURE & STYLE (MANDATORY):\n'
    '1. NEVER use markdown headers (`#`, `##`, `###`). Instead, use **Bold Uppercase Text** '
    'for headings (e.g., **REQUIRED DOCUMENTS**).\n'
    '2. NEVER use horizontal rules (`---`) or emojis.\n'
    '3. If the user asks for a "brief", "short", or "bulleted" response, you MUST '
    'provide NO MORE than 3-5 bullet points total. Do not include an introduction or conclusion.\n'
    '4. Use a clean, professional tone. Avoid generic AI introductory phrases like '
    '"Certainly!" or "Here is the information you requested."\n\n'
    'RESPONSE TEMPLATE EXAMPLE:\n'
    '**SUMMARY**\n'
    '[One or two sentences summarizing the answer]\n\n'
    '**KEY POINTS**\n'
    '- [Point 1]\n'
    '- [Point 2]\n\n'
    '**LEGAL REFERENCE**\n'
    '[Section/Rule citation]'
)

_GEMINI_MODEL = 'gemini-2.5-flash'


def generate_chat_reply(
    message: str,
    retrieved_law: str,
    unique_sources: list[str],
    extracted_text: str | None = None,
    chat_history: list[dict] | None = None,
) -> ChatResponse:
    '''
    Generate a plain-language answer to a tax query using retrieved law context.
    '''
    if not retrieved_law.strip() and not extracted_text:
        return ChatResponse(
            answer='I do not have enough information to answer that.',
            citations=[],
        )

    logger.info('Chat: generating reply for query=\'%s...\'', message[:60])

    prompt = ''
    if retrieved_law.strip():
        prompt += f'Retrieved Legal Context (General Law):\n\n{retrieved_law}\n\n'
        
    if chat_history:
        recent = chat_history[-6:]
        hist_str = '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in recent])
        prompt += f'Recent Chat History:\n{hist_str}\n\n'
        
    if extracted_text:
        prompt += f'Active Document Context (Notice to Analyze - USE THIS FOR SPECIFIC FACTS):\n\n{extracted_text}\n\n'
        
    prompt += f'User Question: {message}'
    messages = [('system', _CHAT_SYSTEM_PROMPT), ('user', prompt)]

    try:
        # 1. Try Primary: AWS Bedrock (Nova Lite)
        logger.info('Chat: Attempting Bedrock LLM')
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        llm_bedrock = ChatBedrockConverse(
            model=settings.bedrock_model_id,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            client=session.client('bedrock-runtime'),
        )
        response = llm_bedrock.invoke(messages)
        answer_text = str(response.content)
    except (BotoCoreError, ClientError) as exc_bedrock:
        logger.warning(
            'Chat: Bedrock API failed for query=\'%s...\': %s', message[:60], exc_bedrock
        )

        # 2. Fallback: Google Gemini Free Tier
        logger.info('Chat: Attempting Gemini LLM fallback')
        try:
            llm_gemini = ChatGoogleGenerativeAI(
                model=_GEMINI_MODEL,
                api_key=settings.gemini_api_key,
                max_output_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
            response = llm_gemini.invoke(messages)
            answer_text = str(response.content)
        except Exception as exc_gemini:
            logger.error(
                'Chat: Both Bedrock and Gemini APIs failed. Bedrock error: %s, Gemini error: %s',
                exc_bedrock,
                exc_gemini,
            )
            raise RuntimeError(
                f'All LLM generation attempts failed. Gemini error: {exc_gemini}'
            ) from exc_gemini

    return ChatResponse(
        answer=answer_text,
        citations=unique_sources,
    )
