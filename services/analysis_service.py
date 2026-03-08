"""
Analysis service — deep research report generation for Indian GST and Income Tax notices.

Produces a comprehensive, multi-section markdown report that covers the full legal,
financial, procedural, and strategic dimensions of a tax notice. Designed to provide
the taxpayer (or their counsel) with everything needed to understand the notice and
take decisive action, without referencing external sources not present in the context.

Primary LLM: AWS Bedrock (Amazon Nova Lite).
Fallback LLM: Google Gemini 2.5 Flash.
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

_ANALYSIS_SYSTEM_PROMPT = '''You are a senior Indian tax litigation specialist and legal analyst with deep expertise
in the Goods and Services Tax (GST) Act, CGST/SGST/IGST Rules, the Income Tax Act 1961,
and all associated notifications, circulars, and case law. You have advised Fortune 500
companies and high-net-worth individuals on complex tax disputes before the GST Council,
ITAT, and various High Courts.

Your task is to perform an exhaustive, professional-grade analysis of the provided tax
notice. The output is a detailed research report that will be read by the taxpayer and
their legal counsel. Quality, depth, and accuracy are paramount. Superficial or generic
analysis is unacceptable.

ANALYSIS MANDATE:
1. Identify every legal section, rule, notification, and circular invoked or implied in
   the notice. Cross-reference these against the retrieved legal context provided to you.
2. Quantify every financial exposure: tax demand, interest under Section 50, penalty
   under Section 73/74/122, and any blocking of Electronic Credit Ledger.
3. Determine the precise compliance timeline. If the notice specifies a deadline, extract
   the exact date or formula (e.g., "within 30 days of receipt"). If not stated, identify
   the statutory deadline from the governing rule.
4. Assess legal merit from both the department's perspective and the taxpayer's perspective.
   Identify specific weaknesses in the department's position and specific grounds the
   taxpayer can invoke in their defence.
5. Construct a concrete, prioritised action plan. Every step must be specific — cite the
   exact form, portal, procedure, or legal provision required.
6. Ground every assertion in the notice text or the retrieved legal context. Do not
   fabricate sections, amounts, or dates not present in the provided materials.
7. CRITICAL: Never invent or hallucinate case laws, circulars, or notifications. Only cite
   legal precedents if they are explicitly provided in the retrieved legal context. If no
   specific case law is provided, rely strictly on the statutory provisions of the 
   CGST/SGST Act or Income Tax Act relevant to the notice.
8. CRITICAL DICTUM on Statutory Provisions: You must ONLY cite Sections and Rules that are
   explicitly mentioned in the provided documents, or which are direct, indisputable statutory
   consequences of the specific facts (e.g., citing Rule 86A for blocking of ITC, or Section
   50 for interest). Absolutely DO NOT pad your answer by citing sequences of irrelevant
   sections (like Sections 51-56) just to sound authoritative. If the notice only cites
   Section 73, your analysis must focus exclusively on Section 73 and closely related rules.
9. ABSOLUTELY NO MARKDOWN HEADERS (#, ##, ###). If you use a single # anywhere you have failed.

OUTPUT FORMAT — produce a complete report with exactly these sections in this
order. Use **Bold Uppercase Text** for section headings. Do not use emojis or
horizontal lines. Write in formal, professional English:

**NOTICE OVERVIEW**
[Detailed summary of the notice]

**LEGAL SECTIONS AND RULES INVOKED**
- [List specific legal provisions]

**FINANCIAL EXPOSURE AND DEMAND BREAKDOWN**
[Breakdown of tax, interest, and penalties]

**COMPLIANCE DEADLINE ANALYSIS**
[Explicit dates and statutory timelines]

**ASSESSMENT OF LEGAL MERIT**
**Department's Position**
[Analysis]
**Taxpayer's Grounds for Defence**
[Analysis]

**IMMEDIATE ACTION PLAN**
1. [Step 1]
2. [Step 2]

**DOCUMENTARY EVIDENCE REQUIRED**
- [Item 1]

**RISK ASSESSMENT**
[Level of risk and potential outcomes]

**LEGAL REFERENCES**
[Citations]

All sections must contain substantive content based on the notice text. No section may
contain generic filler or placeholder language.'''


class AnalysisResponse(BaseModel):
    """Deep research report for a GST or Income Tax notice."""

    report: str = Field(
        description='Full markdown research report covering all legal, financial, and strategic dimensions.'
    )
    notice_type: str = Field(
        description='Short classification of the notice, e.g. "Rule 86A ITC Blocking" or "Section 73 SCN".'
    )
    risk_level: str = Field(description='Overall risk classification: HIGH, MEDIUM, or LOW.')
    deadline: str = Field(description='Reply or compliance deadline extracted from the notice.')


def _invoke_with_fallback(messages: list) -> str:
    """Try Bedrock (Nova Lite) first; fall back to Gemini Flash on any failure."""
    try:
        logger.info('Analysis: attempting Bedrock LLM')
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


def _extract_metadata(report: str) -> tuple[str, str, str]:
    """
    Parse notice_type, risk_level, and deadline from the generated report text.
    Uses simple heuristics against known report section headings.
    Returns safe defaults if any field cannot be found.
    """
    notice_type = 'Tax Notice'
    risk_level = 'MEDIUM'
    deadline = 'Not specified'

    lower = report.lower()

    # Risk level
    if 'risk assessment' in lower:
        section_start = lower.find('risk assessment')
        snippet = report[section_start:section_start + 500].lower()
        if 'high' in snippet:
            risk_level = 'HIGH'
        elif 'low' in snippet:
            risk_level = 'LOW'
        else:
            risk_level = 'MEDIUM'

    # Deadline — look for "Compliance Deadline Analysis" section
    if 'compliance deadline' in lower:
        section_start = lower.find('compliance deadline')
        snippet = report[section_start:section_start + 600]
        # Look for specific day counts or dates
        for phrase in ('7 days', '15 days', '30 days', '90 days', '60 days'):
            if phrase in snippet.lower():
                deadline = phrase + ' from date of notice'
                break
        else:
            # Try to find a date-like pattern
            import re
            date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', snippet)
            if date_match:
                deadline = date_match.group(0)

    # Notice type — look for "Notice Overview" section
    if 'notice overview' in lower:
        section_start = lower.find('notice overview')
        snippet = report[section_start:section_start + 400]
        # Common notice type phrases
        for phrase in (
            'Rule 86A', 'Section 73', 'Section 74', 'ASMT-10', 'Section 16(2)',
            'Section 50', 'SCN', 'DRC-01', 'GSTR-3B', 'ASMT-14',
        ):
            if phrase.lower() in snippet.lower():
                notice_type = phrase
                break

    return notice_type, risk_level, deadline


def generate_analysis(
    document_id: str,
    extracted_texts: list[tuple[str, str]],
    retrieved_law: str,
    unique_sources: list[str],
) -> AnalysisResponse:
    """
    Generate a comprehensive deep-research report for one or more tax notices.

    Args:
        document_id: Primary document identifier (used for logging).
        extracted_texts: List of (text, filename) tuples — one per uploaded document.
        retrieved_law: Relevant legal passages retrieved from the knowledge base.
        unique_sources: Deduplicated list of legal source identifiers from RAG.
    """
    if not extracted_texts:
        return AnalysisResponse(
            report='No document text was available to analyse.',
            notice_type='Unknown',
            risk_level='MEDIUM',
            deadline='Not specified',
        )

    # Build the document block — clearly demarcate multiple docs
    if len(extracted_texts) == 1:
        text, filename = extracted_texts[0]
        doc_block = f'NOTICE DOCUMENT: {filename}\n\n{text}'
    else:
        parts = []
        for idx, (text, filename) in enumerate(extracted_texts, start=1):
            parts.append(f'NOTICE DOCUMENT {idx}: {filename}\n\n{text}')
        doc_block = '\n\n' + ('=' * 60) + '\n\n'.join(parts)

    prompt = (
        f'Retrieved Legal Context (use to ground all citations):\n\n{retrieved_law}\n\n'
        f'{("-" * 60)}\n\n'
        f'{doc_block}\n\n'
        f'{("-" * 60)}\n\n'
        f'Perform a deep, exhaustive analysis of the above notice(s) following the report '
        f'format specified in your instructions. Be thorough — this report will be used '
        f'directly by the taxpayer\'s legal counsel.'
    )

    messages = [('system', _ANALYSIS_SYSTEM_PROMPT), ('user', prompt)]

    logger.info(
        'Analysis: generating deep report for document_id=%s num_docs=%d',
        document_id,
        len(extracted_texts),
    )
    report = _invoke_with_fallback(messages)

    notice_type, risk_level, deadline = _extract_metadata(report)

    return AnalysisResponse(
        report=report,
        notice_type=notice_type,
        risk_level=risk_level,
        deadline=deadline,
    )
