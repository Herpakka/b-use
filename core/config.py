"""Env-driven settings. Every field is read from ``BU_<NAME>`` (or ``.env``); no secrets in code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BU_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # --- LLM
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("BU_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    model_decide: str = "gpt-4.1"
    model_small: str = "gpt-4.1-mini"
    # USD per 1M tokens as (input, output); merged over the defaults in core/costs.py.
    model_prices: dict[str, tuple[float, float]] = Field(default_factory=dict)
    llm_timeout_s: float = 60

    # --- infra
    database_url: str = "postgresql://bu:bu@localhost:5432/bu"
    redis_url: str = "redis://localhost:6379/0"

    # --- browser
    headless: bool = True
    max_contexts: int = 4
    pool_acquire_timeout_s: float = 30
    idle_timeout_s: float = 600
    reaper_interval_s: float = 60
    block_resources: bool = True
    sessions_dir: Path = Path(".data/sessions")
    viewport_width: int = 1280
    viewport_height: int = 800
    nav_timeout_ms: int = 20_000
    action_timeout_ms: int = 5_000
    # Elements/text this many px above/below the viewport are still included in an observation.
    viewport_expansion_px: int = 400
    max_elements: int = 250
    extract_max_chars: int = 20_000

    # --- agent run limits (the three loop guards)
    max_steps: int = 25
    budget_usd: float = 0.15
    run_timeout_s: float = 300

    # --- agent context strategy
    observation_max_chars: int = 12_000
    history_window: int = 20
    findings_window: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
