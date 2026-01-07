from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env."""

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4.1-mini", description="OpenAI model used for vision analysis.")
    frame_samples: int = Field(20, description="Minimum number of frames to sample from each video.")
    seconds_per_frame: float = Field(
        2.0, description="Target sampling interval in seconds (more frames for longer videos)."
    )
    max_frame_samples: int = Field(
        120, description="Upper bound on frames sent to the model to stay within payload limits."
    )
    max_tokens: int = Field(5000, description="Max tokens to request from the model for richer responses.")

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
