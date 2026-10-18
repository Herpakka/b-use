"""Env-driven settings. Every field is read from ``BU_<NAME>`` (or ``.env``); no secrets in code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StaticSecret(BaseModel):
    value: SecretStr
    domains: list[str] = Field(default_factory=list)


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
    # "redis" shares live events and rate-limit counters across processes; "local" is one process.
    event_bus: Literal["redis", "local"] = "redis"

    # --- API
    # API key → tenant, as JSON: BU_API_KEYS={"<long random key>": "acme"}. Empty = API refuses.
    api_keys: dict[str, str] = Field(default_factory=dict)
    rate_limit_per_minute: int = 120
    max_active_runs_per_tenant: int = 3
    max_budget_usd: float = 1.0  # cap on a per-run budget a client may request
    sse_heartbeat_s: float = 15
    debug_page: bool = True

    # --- secrets: {{SECRET:name}} placeholders (core/secrets.py)
    # 32 random bytes, base64; encrypts tenants' secrets at rest. Unset = the secrets API is off.
    secrets_key: SecretStr | None = None
    # For the CLI and evals only — the API keeps per-tenant secrets in Postgres. JSON:
    # BU_LOCAL_SECRETS={"shop_password": {"value": "…", "domains": ["shop.example"]}}
    local_secrets: dict[str, StaticSecret] = Field(default_factory=dict)

    # --- spend caps in USD per calendar month (UTC). A run starts only if its whole budget fits
    # in what is left, counting active runs at their full budget — so the cap is never exceeded
    # by more than one LLM call per run.
    monthly_budget_usd: float | None = None  # all tenants together
    tenant_monthly_budget_usd: float | None = None  # each tenant

    # --- privacy and retention
    # Mask e-mail addresses, phone/card/national-ID numbers and API keys in logs and run events.
    redact_pii: bool = True
    screenshot_ttl_s: float = 86_400
    # Agent checkpoints hold page snapshots; a finished run no longer needs them.
    checkpoint_ttl_s: float = 86_400
    run_retention_days: float | None = None  # then delete the run and its events too
    janitor_interval_s: float = 3600

    # --- browser
    # Drive a browser served elsewhere instead of launching one, e.g. the visible Chromium from
    # `python -m devtools.browser_host`: ws://host.docker.internal:3010/<token>
    browser_ws_endpoint: str | None = None
    # Every browser request goes through the egress proxy (browser/egress.py), which refuses
    # private/internal addresses — including redirect hops the request router never sees.
    # With a local browser the pool runs it on 127.0.0.1; a remote browser uses the one that
    # devtools.browser_host runs beside it (password = the endpoint's token).
    egress_proxy: bool = True
    egress_listen_host: str = "127.0.0.1"
    egress_listen_port: int = 0
    # The proxy URL as the *browser* reaches it. None = the local proxy's address, or
    # http://127.0.0.1:3129 (browser_host's default) for a remote browser.
    egress_browser_url: str | None = None
    headless: bool = True
    max_contexts: int = 4
    pool_acquire_timeout_s: float = 30
    idle_timeout_s: float = 600
    # Sessions whose run waits for a human (login, captcha, approval) are kept this long.
    held_idle_timeout_s: float = 7200
    reaper_interval_s: float = 60
    # A browser that stops answering this often/this long is restarted (sessions are restored
    # by the agent from their cookies and last URL).
    browser_probe_interval_s: float = 30
    browser_probe_timeout_s: float = 10
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

    @field_validator("browser_ws_endpoint", "egress_browser_url", "secrets_key", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        # `BU_BROWSER_WS_ENDPOINT=` must mean "no remote browser" — an empty string would
        # otherwise select remote mode and leave the browser without its local egress proxy.
        return None if isinstance(v, str) and not v.strip() else v


@lru_cache
def get_settings() -> Settings:
    return Settings()
