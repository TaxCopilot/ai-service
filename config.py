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
    aws_region: str = Field(default='us-east-1')

    # Bedrock — LLM
    bedrock_model_id: str = Field(default='anthropic.claude-sonnet-4-6')
    bedrock_max_tokens: int = Field(default=2048)
    # Low temp keeps legal drafts factual and consistent
    bedrock_temperature: float = Field(default=0.2)

    # Bedrock — Knowledge Bases
    bedrock_knowledge_base_id: str = Field(default='PLACEHOLDER_KB_ID')
    bedrock_retrieval_results: int = Field(default=5)

    # Textract
    textract_max_pages: int = Field(default=15)  # cost guard

    # Server
    api_host: str = Field(default='0.0.0.0')
    api_port: int = Field(default=8001)
    log_level: str = Field(default='INFO')


settings = Settings()
