# core/security.py
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional

from core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not settings.API_KEY_ENABLED:
        return None
    
    if api_key == settings.API_KEY:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API Key"
    )# API key authentication (optional)
