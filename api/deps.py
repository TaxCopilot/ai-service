"""FastAPI dependencies shared across routes."""

import logging

from fastapi import Header, HTTPException, status

from config import settings

logger = logging.getLogger(__name__)

_AUTH_DISABLED_MSG = (
    'API_KEY is not set — X-API-Key authentication is disabled. '
    'Set API_KEY in .env before deploying.'
)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Verify the X-API-Key header matches the configured secret.

    If API_KEY is not set in config (dev mode), the check is skipped
    entirely and a startup warning is emitted instead.
    """
    if settings.api_key is None:
        # Auth is intentionally disabled — warn once (at startup via main.py).
        return

    if x_api_key != settings.api_key:
        logger.warning('Rejected request with invalid or missing X-API-Key')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or missing X-API-Key header.',
        )
