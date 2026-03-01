'App config — all values come from environment variables (or .env).\nImport `settings` everywhere; never instantiate Settings() directly.\n'

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

    # AWS
    aws_region: str = Field(default='us-east-1', alias='AWS_DEFAULT_REGION')
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)

    # Shared secret expected in the X-API-Key header from the TypeScript backend.
    # Leave unset (None) during local development to disable the check.
    api_key: str | None = Field(default=None)

    # Bedrock — LLM
    bedrock_model_id: str = Field(default='anthropic.claude-sonnet-4-6')
    bedrock_max_tokens: int = Field(default=2048)
    # Low temp keeps legal drafts factual and consistent
    bedrock_temperature: float = Field(default=0.2)
    # Characters of notice text sent to the LLM — prevents context overflow on long notices
    bedrock_max_notice_chars: int = Field(default=15_000)


    # Bedrock — Knowledge Bases (Legacy)
    bedrock_knowledge_base_id: str = Field(default='PLACEHOLDER_KB_ID')
    bedrock_retrieval_results: int = Field(default=5)

    # PostgreSQL / pgvector Database
    database_url: str = Field(default='postgresql://postgres:postgres@localhost:5432/postgres')

    # Textract
    textract_max_pages: int = Field(default=15)  # cost guard
    # Scanned Indian tax notices (fax, photocopy) typically score 60-75%
    # in Textract; 80 dropped legitimate text. 50 keeps readable content
    # while still filtering pure noise.
    textract_min_confidence: float = Field(default=50.0)


    # Server
    api_host: str = Field(default='0.0.0.0')
    api_port: int = Field(default=8001)
    log_level: str = Field(default='INFO')


settings = Settings()
