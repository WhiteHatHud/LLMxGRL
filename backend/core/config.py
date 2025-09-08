# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "GraphLLM-Backend"
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./graphllm.db"
    
    # LLM settings
    LLM_PROVIDER: str = "openrouter"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    LLM_TEMPERATURE: float = 0.2
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 512
    LLM_ENDPOINT_BASE: str = "https://api.openrouter.ai/v1"
    
    # RAG defaults
    RETRIEVER_TOP_K: int = 5
    RETRIEVER_HOPS: int = 2
    PROMPT_TEMPLATE_ID: str = "default_rag"
    
    # Security
    API_KEY_ENABLED: bool = False
    API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()# Settings loaded from .env
