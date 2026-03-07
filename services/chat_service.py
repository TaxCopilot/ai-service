'''
Chat service for answering general tax-law questions using RAG + Gemini.

This is the default 'chat' mode — the user asks a freeform question and
Gemini answers using passages retrieved from the tax_laws vector store.
No document upload or OCR required.
'''

import logging

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


_CHAT_SYSTEM_PROMPT = '''
You are TaxCopilot, a helpful and precise Indian tax law assistant.
Your role is to answer questions, analyze notices, and draft replies about Indian GST and Income Tax.

RULES:
1. Base your answer on the retrieved legal context, the Active Document Context, and Recent Chat History.
2. If the user asks a legal question that cannot be answered from the provided context, state that you do not have enough information. However, if the user asks you to draft a reply, summarize text, or analyze the active document, you MUST fulfill the request using the provided document context.
3. Do NOT fabricate section numbers, rules, or dates.
4. Use plain, professional language. Avoid excessive legal jargon.
5. If answering a legal query, cite the specific section or rule you are referencing at the end of your answer.
'''

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
    if extracted_text:
        prompt += f'Active Document Context (Notice to Analyze):\n\n{extracted_text}\n\n'
    if chat_history:
        recent = chat_history[-6:]
        hist_str = '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in recent])
        prompt += f'Recent Chat History:\n{hist_str}\n\n'
        
    prompt += f'Retrieved Legal Context:\n\n{retrieved_law}\n\nUser Question: {message}'
    messages = [('system', _CHAT_SYSTEM_PROMPT), ('user', prompt)]

    try:
        # 1. Try Primary: AWS Bedrock (Nova Lite)
        logger.info('Chat: Attempting Bedrock LLM')
        llm_bedrock = ChatBedrockConverse(
            model=settings.bedrock_model_id,
            region_name=settings.aws_region,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
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
