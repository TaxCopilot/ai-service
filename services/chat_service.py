'''
Chat service for answering general tax-law questions using RAG + Gemini.

This is the default "chat" mode — the user asks a freeform question and
Gemini answers using passages retrieved from the tax_laws vector store.
No document upload or OCR required.
'''

import logging

from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
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
Your role is to answer questions about Indian GST, Income Tax, and related regulations.

RULES:
1. ONLY answer using the retrieved legal context provided below.
2. If the context does not contain enough information, say: "I don't have enough information in my legal database to answer that with confidence."
3. Do NOT fabricate section numbers, rules, or dates.
4. Use plain, professional language. Avoid excessive legal jargon.
5. Cite the specific section or rule you are referencing at the end of your answer.
'''


def generate_chat_reply(
    message: str,
    retrieved_law: str,
    unique_sources: list[str],
) -> ChatResponse:
    '''
    Generate a plain-language answer to a tax query using retrieved law context.
    '''
    if not retrieved_law.strip():
        return ChatResponse(
            answer='I don\'t have enough information in my legal database to answer that with confidence.',
            citations=[],
        )

    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        api_key=settings.gemini_api_key,
        max_output_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )

    logger.info('Chat: generating reply for query="%s..."', message[:60])

    prompt = f'Retrieved Legal Context:\n\n{retrieved_law}\n\nUser Question: {message}'

    try:
        response = llm.invoke([('system', _CHAT_SYSTEM_PROMPT), ('user', prompt)])
        answer_text = str(response.content)
    except (BotoCoreError, ClientError) as exc:
        logger.error('Chat: Bedrock API failed for query="%s...": %s', message[:60], exc)
        raise RuntimeError(f'Bedrock LLM generation failed: {exc}') from exc

    return ChatResponse(
        answer=answer_text,
        citations=unique_sources,
    )
