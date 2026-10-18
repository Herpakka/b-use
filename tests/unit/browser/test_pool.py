import asyncio
from pathlib import Path

import pytest

from browser.pool import BrowserPool, PoolExhaustedError
from core.config import Settings
from tests.browser_utils import FIXTURE_DOMAINS, goto, serve_fixtures
from tests.conftest import make_settings

pytestmark = pytest.mark.browser


async def test_storage_state_persists_across_sessions(pool: BrowserPool) -> None:
    s = await pool.open_session("persist-me", allowed_domains=FIXTURE_DOMAINS)
    await serve_fixtures(s.context)
    await goto(s, "static.html")
    page = await s.current_page()
    await page.evaluate("document.cookie = 'login=yes; max-age=3600'; localStorage.k = 'v'")
    await pool.close_session("persist-me")
    assert pool.storage_path("persist-me").exists()

    s = await pool.open_session("persist-me", allowed_domains=FIXTURE_DOMAINS)
    await serve_fixtures(s.context)
    await goto(s, "static.html")
    page = await s.current_page()
    assert await page.evaluate("document.cookie") == "login=yes"
    assert await page.evaluate("localStorage.k") == "v"
    await pool.close_session("persist-me", persist=False)


async def test_open_is_idempotent_per_id(pool: BrowserPool) -> None:
    a, b = await asyncio.gather(
        pool.open_session("same-id", allowed_domains=FIXTURE_DOMAINS),
        pool.open_session("same-id", allowed_domains=FIXTURE_DOMAINS),
    )
    assert a is b
    await pool.close_session("same-id", persist=False)


async def test_invalid_session_id_rejected(pool: BrowserPool) -> None:
    with pytest.raises(ValueError):
        await pool.open_session("../../etc/passwd", allowed_domains=[])


async def test_reaper_closes_idle_sessions(pool: BrowserPool) -> None:
    s = await pool.open_session("idle-one", allowed_domains=FIXTURE_DOMAINS)
    busy = await pool.open_session("busy-one", allowed_domains=FIXTURE_DOMAINS)
    far_future = s.last_used + pool.settings.idle_timeout_s + 1
    async with busy.lock:  # mid-action sessions are never reaped
        reaped = await pool.reap_idle(now=far_future)
    assert "idle-one" in reaped and "busy-one" not in reaped
    assert pool.get("idle-one") is None and s.closed
    await pool.close_session("busy-one", persist=False)


async def test_pool_exhaustion(tmp_path: Path) -> None:
    settings: Settings = make_settings(tmp_path, max_contexts=1, pool_acquire_timeout_s=0.3)
    try:
        small = BrowserPool(settings)
        await small.start()
    except Exception as err:
        pytest.skip(f"Chromium unavailable: {err}")
    try:
        await small.open_session("one", allowed_domains=[])
        with pytest.raises(PoolExhaustedError):
            await small.open_session("two", allowed_domains=[])
        await small.close_session("one", persist=False)
        await small.open_session("two", allowed_domains=[])  # slot was released
    finally:
        await small.stop()


async def test_held_sessions_get_the_longer_idle_timeout(pool: BrowserPool) -> None:
    held = await pool.open_session("waiting-human", allowed_domains=FIXTURE_DOMAINS)
    plain = await pool.open_session("idle-plain", allowed_domains=FIXTURE_DOMAINS)
    held.held = True
    after_idle = max(held.last_used, plain.last_used) + pool.settings.idle_timeout_s + 1
    assert await pool.reap_idle(now=after_idle) == ["idle-plain"]
    assert pool.get("waiting-human") is held
    after_held = held.last_used + pool.settings.held_idle_timeout_s + 1
    assert await pool.reap_idle(now=after_held) == ["waiting-human"]
