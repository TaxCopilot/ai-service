'''
TaxCopilot AI Microservice entry point.

Run locally:
    uvicorn main:app --reload --port 8001
'''


import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from config import settings
from services.db_service import ensure_table


logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        'Starting up | region=%s model=gemini db=%s',
        settings.aws_region,
        settings.database_url.split('@')[-1] if '@' in settings.database_url else 'local',
    )

    if settings.aws_access_key_id:
        logger.info('AWS credentials loaded: %s...', settings.aws_access_key_id[:8])
    else:
        logger.warning('AWS credentials NOT loaded — check .env file')

    if settings.api_key is None:
        logger.warning('API_KEY is not set — X-API-Key authentication is DISABLED.')

    try:
        ensure_table()
    except RuntimeError as exc:
        logger.error(
            'document_cache table could not be created: %s — '
            'decode mode will fail on cache operations.',
            exc,
        )

    yield
    logger.info('Shutting down.')


app = FastAPI(
    title='TaxCopilot AI Microservice',
    description='Analyses Indian income tax notices using Textract, Bedrock Knowledge Bases, and Claude.',
    version='1.0.0',
    lifespan=lifespan,
)

_MAX_BODY_SIZE_BYTES = 10 * 1024  # 10KB


@app.middleware('http')
async def limit_body_size(request: Request, call_next):
    if 'content-length' in request.headers:
        if int(request.headers['content-length']) > _MAX_BODY_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={'detail': 'Request body exceeds 10KB limit.'},
            )
    return await call_next(request)

# Permissive CORS for the hackathon — restrict allow_origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['POST', 'GET', 'OPTIONS'],
    allow_headers=['*'],
)

app.include_router(router)


@app.get('/health', tags=['ops'])
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'taxcopilot-ai', 'version': '1.0.0'}


if __name__ == '__main__':
    uvicorn.run('main:app', host=settings.api_host, port=settings.api_port, reload=True)
