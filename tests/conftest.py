from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from browser.pool import BrowserPool
from browser.session import BrowserSession
from core.config import Settings
from tests.browser_utils import FIXTURE_DOMAINS, serve_fixtures


class IsolatedSettings(Settings):
    """Settings from explicit arguments only. Neither the environment nor .env may leak in:
    pydantic-settings deep-merges dict fields across sources, and inside the container .env
    arrives as environment variables (e.g. BU_BROWSER_WS_ENDPOINT would hijack the tests)."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def make_settings(tmp: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "env": "test",
        "sessions_dir": tmp,
        "reaper_interval_s": 3600,
        **overrides,
    }
    return IsolatedSettings.model_validate(values)


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return make_settings(tmp_path_factory.mktemp("sessions"))


@pytest.fixture(scope="session")
async def pool(test_settings: Settings) -> AsyncIterator[BrowserPool]:
    p = BrowserPool(test_settings)
    try:
        await p.start()
    except Exception as err:  # Chromium not installed (e.g. on the Windows host)
        pytest.skip(f"Chromium unavailable: {str(err).splitlines()[0][:120]}")
    yield p
    await p.stop()


@pytest.fixture
async def session(pool: BrowserPool) -> AsyncIterator[BrowserSession]:
    sid = f"t-{secrets.token_hex(4)}"
    s = await pool.open_session(sid, allowed_domains=FIXTURE_DOMAINS)
    await serve_fixtures(s.context)
    yield s
    await pool.close_session(sid, persist=False)
