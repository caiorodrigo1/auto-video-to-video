from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    anthropic_api_key: str
    openai_api_key: str
    pexels_api_key: str
    pixabay_api_key: str
    unsplash_api_key: str

    # LLM
    default_llm_provider: Literal["claude", "gpt"] = "claude"
    claude_model: str = "claude-sonnet-4-6"
    gpt_model: str = "gpt-5"

    # Pipeline
    target_block_duration: float = 8.0
    target_resolution: tuple[int, int] = (1920, 1080)
    target_fps: int = 30
    clip_audio_db_offset: float = -20.0

    # Selector weights
    weight_aspect: float = 1.0
    weight_duration: float = 1.0
    weight_resolution: float = 1.0
    weight_source: float = 1.0
    image_montage_max: int = 3

    # Search
    search_top_k: int = 8
    search_concurrency: int = 5

    # Ken Burns
    ken_burns_zoom_start: float = 1.0
    ken_burns_zoom_end: float = 1.15
    ken_burns_pan_pct: float = 0.05

    # Loop crossfade
    loop_crossfade_seconds: float = 0.3

    # Paths
    runs_dir: str = "runs"
    cache_dir: str = "cache"
