from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from browser.pool import BrowserPool
from browser.session import BrowserSession
from core.config import Settings
from tests.browser_utils import FIXTURE_DOMAINS, serve_fixtures


def make_settings(tmp: Path, **overrides: object) -> Settings:
    """Defaults + overrides only: model_validate skips the env/.env sources entirely."""
    values: dict[str, object] = {
        "env": "test",
        "sessions_dir": tmp,
        "reaper_interval_s": 3600,
        **overrides,
    }
    return Settings.model_validate(values)


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
