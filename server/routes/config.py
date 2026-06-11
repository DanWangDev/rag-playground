"""Config route — read and update runtime configuration.

GET  /api/config — Current model, strategy, and parameters
POST /api/config — Update runtime settings
"""

from fastapi import APIRouter
from pydantic import BaseModel

from rag_playground.config.env import settings as app_settings

router = APIRouter(prefix="/api", tags=["config"])


class ConfigResponse(BaseModel):
    chat_model: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    ollama_host: str


class ConfigUpdate(BaseModel):
    top_k: int | None = None
    temperature: float | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    return ConfigResponse(
        chat_model=app_settings.chat_model,
        embed_model=app_settings.embed_model,
        chunk_size=app_settings.chunk_size,
        chunk_overlap=app_settings.chunk_overlap,
        top_k=app_settings.top_k,
        temperature=app_settings.temperature,
        ollama_host=app_settings.ollama_host,
    )


@router.post("/config", response_model=ConfigResponse)
async def update_config(update: ConfigUpdate):
    """Update runtime configuration (non-persistent)."""
    if update.top_k is not None:
        app_settings.top_k = update.top_k
    if update.temperature is not None:
        app_settings.temperature = update.temperature
    if update.chunk_size is not None:
        app_settings.chunk_size = update.chunk_size
    if update.chunk_overlap is not None:
        app_settings.chunk_overlap = update.chunk_overlap

    return await get_config()
