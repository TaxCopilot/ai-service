'App config — all values come from environment variables (or .env).\nImport `settings` everywhere; never instantiate Settings() directly.\n'

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

    # AWS
    aws_region: str = Field(default='ap-south-1', alias='AWS_DEFAULT_REGION')
    aws_access_key_id: str | None = Field(default=None, alias='AWS_ACCESS_KEY_ID')
    aws_secret_access_key: str | None = Field(default=None, alias='AWS_SECRET_ACCESS_KEY')

    # Shared secret expected in the X-API-Key header from the TypeScript backend.
    # Leave unset (None) or empty during local development to disable the check.
    api_key: str | None = Field(default=None)

    @field_validator('api_key', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        if v == '' or v is None:
            return None
        return v

    # Gemini — LLM
    gemini_api_key: str = Field(default='...', alias='GEMINI_API_KEY')
    llm_max_tokens: int = Field(default=8192)
    # Low temp keeps legal drafts factual and consistent
    llm_temperature: float = Field(default=0.2)
    min_notice_chars: int = Field(default=15_000)


    # Bedrock — Knowledge Bases (Legacy)
    bedrock_model_id: str = Field(default='global.amazon.nova-2-lite-v1:0')
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
    api_port: int = Field(default=8002)
    log_level: str = Field(default='INFO')
    cors_origins: str = Field(default='*')


settings = Settings()
