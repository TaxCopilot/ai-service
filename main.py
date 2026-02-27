"""
TaxCopilot AI Microservice entry point.

Run locally:
    uvicorn main:app --reload --port 8001
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import settings

logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting up | region=%s model=%s kb=%s",
                settings.aws_region, settings.bedrock_model_id, settings.bedrock_knowledge_base_id)

    if settings.bedrock_knowledge_base_id == "PLACEHOLDER_KB_ID":
        logger.warning("BEDROCK_KNOWLEDGE_BASE_ID is not set — requests will fail until configured.")

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="TaxCopilot AI Microservice",
    description="Analyses Indian income tax notices using Textract, Bedrock Knowledge Bases, and Claude.",
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS for the hackathon — restrict allow_origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "service": "taxcopilot-ai", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
