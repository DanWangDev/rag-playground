"""Shared Pydantic validators used across modules.

Centralizes common validation logic so it's defined once and reused.
All user-facing inputs pass through these schemas before touching
any pipeline code.

Key Learning Points:
  - Pydantic provides runtime type checking + clear error messages
  - Shared validators prevent duplication across modules
  - Pydantic models can be used both for validation and as typed data containers
"""

from pydantic import BaseModel, Field


class PositiveInt(BaseModel):
    """A positive integer (> 0)."""

    value: int = Field(gt=0)


class NonNegativeInt(BaseModel):
    """A non-negative integer (>= 0)."""

    value: int = Field(ge=0)


class NonEmptyStr(BaseModel):
    """A non-empty string."""

    value: str = Field(min_length=1)


class Temperature(BaseModel):
    """Model temperature (0.0 to 2.0)."""

    value: float = Field(ge=0.0, le=2.0)


class TopK(BaseModel):
    """Number of results to retrieve (1 to 100)."""

    value: int = Field(ge=1, le=100)


class ChunkSize(BaseModel):
    """Chunk size in characters (100 to 8000)."""

    value: int = Field(ge=100, le=8000)


class ChunkOverlap(BaseModel):
    """Chunk overlap in characters (0 to 2000)."""

    value: int = Field(ge=0, le=2000)
