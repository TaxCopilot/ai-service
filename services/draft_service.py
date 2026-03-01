import logging
import re

from langchain_google_genai import ChatGoogleGenerativeAI
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
         
    for cite in citations:
        # Use simple lowercase containment check for robustness
        clean_cite = cite.lower().strip()
        if clean_cite not in retrieved_law.lower():
            logger.warning(f'🚨 HALLUCINATION DETECTED: {cite} not in retrieved context.')
            return False
            
    return True

def generate_notice_reply(query: str) -> NoticeResponse:
    '''
    Orchestrates the RAG pipeline to generate a grounded legal reply.
    '''
    law_context = retrieve_relevant_law(query)
    
    if not law_context:
        return NoticeResponse(
            draft_reply='Insufficient information in current legal corpus.',
            citations=[],
            is_grounded=False
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=settings.gemini_api_key,
        max_output_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature
    )

    print(f"DEBUG: Gemini Request -> Model: gemini-2.5-flash")

    prompt = f'Retrieved Legal Context:\n\n{law_context}\n\nUser Query: {query}'
    
    response = llm.invoke([('system', _SYSTEM_PROMPT), ('user', prompt)])
    content = response.content
    
    is_valid = _extract_and_validate_citations(content, law_context)
    
    # If hallucination detected, attempt ONE surgical regeneration
    if not is_valid:
        print('♻️ Hallucination detected. Retrying with strict warnings...')
        warning_msg = (
            'WARNING: Your previous draft cited sections/rules NOT present in the context. '
            'REGENERATE the draft and ONLY cite the legal text provided. '
            'Do NOT fabricate section numbers.'
        )
        retry_prompt = f'{prompt}\n\n{warning_msg}'
        response = llm.invoke([('system', _SYSTEM_PROMPT), ('user', retry_prompt)])
        content = response.content
        
        is_valid = _extract_and_validate_citations(content, law_context)

    if not is_valid:
         return NoticeResponse(
            draft_reply='The generated draft contained invalid citations and was discarded for safety.',
            citations=[],
            is_grounded=False
         )

    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            import json
            data = json.loads(json_match.group(0))
            return NoticeResponse(
                draft_reply=data.get('draft_reply', 'Error parsing draft.'),
                citations=data.get('citations', []),
                is_grounded=True
            )
        except Exception:
            pass

    return NoticeResponse(
        draft_reply=content,
        citations=re.findall(r'(?:Section|Rule)\s+\d+[A-Z]*', content),
        is_grounded=is_valid
    )
