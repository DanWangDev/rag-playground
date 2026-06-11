"""Pydantic-based environment settings.

Loads configuration from .env file with sensible defaults.
All modules import `settings` from here — never read os.environ directly.

Key Learning Points:
  - Pydantic Settings provides validation + type coercion for env vars
  - lru_cache ensures config is parsed once, not on every import
  - Default values mean the app works without any .env file
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Chat Provider ---
    chat_provider: str = "ollama"
    chat_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"

    # --- Embedding Provider ---
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text"
    embed_dimensions: int = 768

    # --- Pipeline Defaults ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    temperature: float = 0.1
    data_dir: str = "./data"
    log_level: str = "info"


settings = Settings()
