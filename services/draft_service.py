import json
import logging
import re

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from pydantic import BaseModel, Field

from config import settings
from services.kb_service import retrieve_relevant_law

logger = logging.getLogger(__name__)


class NoticeResponse(BaseModel):
    '''Response schema for the legal draft.'''
    draft_reply: str = Field(description='The generated reply to the GST tax notice.')
    citations: list[str] = Field(description='List of legal sections or rules cited.')
    is_grounded: bool = Field(description='Whether the response is fully grounded in retrieved law.')

# Hardened System Prompt
_SYSTEM_PROMPT = '''
You are a "Grounded Legal Draft Assistant" providing precise replies to GST tax notices.
Your primary directive is accuracy and grounding in the provided legal corpus.

STRICT GROUNDING RULES:
1. ONLY use the legal context provided below.
2. If the context does not contain enough information to answer the query, respond exactly: 
   "Insufficient information in current legal corpus."
3. Do NOT invent or hallucinate section numbers or legal rules.
4. ONLY cite section or rule numbers that explicitly appear in the retrieved context.
5. Do NOT rely on general legal knowledge or external facts.

OUTPUT FORMAT:
- Your response must be professional and follow standard legal drafting norms.
- Use the following JSON-like structure (which will be parsed into a Pydantic model):
  {
    "draft_reply": "Detailed legal response...",
    "citations": ["Section 73", "Rule 142"],
    "is_grounded": true
  }
'''

def _extract_and_validate_citations(llm_output: str, retrieved_law: str) -> bool:
    '''
    Extracts citations from LLM output and validates them against the retrieved law text.
    Returns True if all citations are found in the retrieved context.
    '''
    # Extract Section/Rule numbers like "Section 73" or "Rule 142"
    citations = re.findall(r'(?:Section|Rule)\s+\d+[A-Z]*', llm_output, flags=re.IGNORECASE)
    
    if not citations:
         return True
         
    for cite in set(citations):
        # Use simple lowercase containment check for robustness
        clean_cite = cite.lower().strip()
        
        # Look for the bare number since the raw text might just say "73. (1)..."
        num_match = re.search(r'\d+[A-Z]*', clean_cite)
        if num_match:
            number = num_match.group(0)
            if number not in retrieved_law.lower():
                logger.warning('🚨 HALLUCINATION DETECTED: %s (number %s) not in retrieved context.', cite, number)
                return False
        elif clean_cite not in retrieved_law.lower():
            logger.warning('🚨 HALLUCINATION DETECTED: %s not in retrieved context.', cite)
            return False
            
    return True

def generate_notice_reply(document_id: str, extracted_text: str, retrieved_law: str, unique_sources: list[str]) -> NoticeResponse:
    '''
    Orchestrates the RAG pipeline to generate a grounded legal reply using the retrieved laws.
    '''
    if not retrieved_law.strip():
        return NoticeResponse(
            draft_reply='Insufficient information in current legal corpus based on the provided text.',
            citations=[],
            is_grounded=False
        )

    try:
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
    except (BotoCoreError, ClientError) as exc:
        logger.error('Draft: failed to initialise Bedrock client for %s: %s', document_id, exc)
        raise RuntimeError(f'LLM client initialisation failed: {exc}') from exc

    logger.info('Drafting reply for document_id=%s using %s', document_id, settings.bedrock_model_id)

    prompt = f'Retrieved Legal Context:\n\n{retrieved_law}\n\nNotice Text to Reply To:\n\n{extracted_text}'
    
    try:
        response = llm.invoke([('system', _SYSTEM_PROMPT), ('user', prompt)])
        content = str(response.content)
    except (BotoCoreError, ClientError) as exc:
        logger.error('Draft: Bedrock API failed for document_id=%s: %s', document_id, exc)
        raise RuntimeError(f'Bedrock LLM generation failed: {exc}') from exc

    is_valid = _extract_and_validate_citations(content, retrieved_law)

    # If hallucination detected, attempt ONE surgical regeneration
    if not is_valid:
        logger.warning('Hallucination detected for %s. Retrying with strict warnings...', document_id)
        warning_msg = (
            'WARNING: Your previous draft cited sections/rules NOT present in the context. '
            'REGENERATE the draft and ONLY cite the legal text provided. '
            'Do NOT fabricate section numbers.'
        )
        retry_prompt = f'{prompt}\n\n{warning_msg}'
        try:
            response = llm.invoke([('system', _SYSTEM_PROMPT), ('user', retry_prompt)])
            content = str(response.content)
        except (BotoCoreError, ClientError) as exc:
            logger.error('Draft: Bedrock API retry failed for document_id=%s: %s', document_id, exc)
            raise RuntimeError(f'Bedrock LLM generation failed on retry: {exc}') from exc
        
        is_valid = _extract_and_validate_citations(content, retrieved_law)

    if not is_valid:
         return NoticeResponse(
            draft_reply='The generated draft contained invalid citations and was discarded for safety.',
            citations=[],
            is_grounded=False
         )

    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))

            # Combine validated sources directly from pg-vector and explicit matched llm citations
            final_citations = list(set(data.get('citations', []) + unique_sources))

            return NoticeResponse(
                draft_reply=data.get('draft_reply', 'Error parsing draft.'),
                citations=final_citations,
                is_grounded=True,
            )
        except json.JSONDecodeError as exc:
            logger.warning('Draft: LLM output was not valid JSON for %s: %s', document_id, exc)

    return NoticeResponse(
        draft_reply=content,
        citations=list(set(re.findall(r'(?:Section|Rule)\s+\d+[A-Z]*', content) + unique_sources)),
        is_grounded=is_valid
    )
