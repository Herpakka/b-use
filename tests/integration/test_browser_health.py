"""Chromium dying — a crashed tab, a killed browser, a frozen one — and the run carrying on."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import psutil
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from playwright.async_api import Error as PlaywrightError

from agent.context import AgentContext
from agent.graph import compile_graph
from agent.runner import drive, initial_state, limits_from
from browser.actions import AnyAction, Click
from browser.local import LocalBrowserService
from browser.models import ActionResult
from browser.pool import BrowserPool, kill_marked
from browser.session import BrowserSession
from core.costs import PriceTable
from core.secrets import Secret
from tests.agent_fakes import PLAN, ScriptedChatModel, tool
from tests.browser_utils import FIXTURE_DOMAINS, goto, serve_fixtures
from tests.conftest import make_settings

pytestmark = pytest.mark.browser


def browser_processes(pool: BrowserPool) -> list[psutil.Process]:
    out = []
    for proc in psutil.process_iter(["cmdline"]):
        if pool.marker in (proc.info["cmdline"] or []):
            out.append(proc)
    return out


async def until(predicate: object, within_s: float = 10.0) -> None:
    assert callable(predicate)
    deadline = asyncio.get_running_loop().time() + within_s
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.05)


@pytest.fixture
async def own_pool(tmp_path: Path) -> AsyncIterator[BrowserPool]:
    """A pool of its own: these tests kill its browser."""
    pool = BrowserPool(make_settings(tmp_path, browser_probe_timeout_s=1.0))
    try:
        await pool.start()
    except Exception as err:
        pytest.skip(f"Chromium unavailable: {err}")
    yield pool
    await pool.stop()


async def open_fixture_session(pool: BrowserPool, sid: str) -> BrowserSession:
    session = await pool.open_session(sid, allowed_domains=FIXTURE_DOMAINS)
    await serve_fixtures(session.context)
    return session


async def test_crashed_tab_is_reopened_on_the_same_page(session: BrowserSession) -> None:
    await goto(session, "static.html")
    page = await session.current_page()
    with pytest.raises(PlaywrightError):
        await page.goto("chrome://crash", timeout=5000)
    await until(lambda: page in session._crashed)

    obs = await session.observe()
    assert obs.url == "https://fixtures.test/static.html"
    assert "Notice: the tab crashed and was reopened at https://fixtures.test/static.html" in (
        obs.render()
    )
    assert obs.elements and obs.tabs == 1


async def test_killed_browser_is_relaunched_and_logins_survive(own_pool: BrowserPool) -> None:
    s = await open_fixture_session(own_pool, "crash-me")
    await goto(s, "static.html")
    page = await s.current_page()
    await page.evaluate("document.cookie = 'login=yes; max-age=3600'")
    await own_pool.checkpoint_storage()  # what the probe does every interval

    assert kill_marked(own_pool.marker) >= 1
    await until(lambda: own_pool.get("crash-me") is None)  # dropped on disconnect
    assert not own_pool.health().ok and s.closed

    await own_pool.probe_once()  # relaunches right away
    assert own_pool.health().ok
    s = await open_fixture_session(own_pool, "crash-me")
    await goto(s, "static.html")
    assert await (await s.current_page()).evaluate("document.cookie") == "login=yes"


async def test_frozen_browser_is_killed_and_replaced(own_pool: BrowserPool) -> None:
    await open_fixture_session(own_pool, "frozen")
    main = next(p for p in browser_processes(own_pool) if "--type=" not in " ".join(p.cmdline()))
    main.suspend()  # SIGSTOP: alive, but answers nothing
    try:
        assert await own_pool.probe() == "no answer within 1s"
        await own_pool.probe_once()  # one miss is tolerated
        assert own_pool.restarts == 0
        await own_pool.probe_once()
        assert own_pool.restarts == 1 and own_pool.health().ok
        assert own_pool.get("frozen") is None
        await asyncio.sleep(0.5)
        assert not main.is_running() or main.status() == psutil.STATUS_ZOMBIE
    finally:
        if main.is_running():
            main.kill()
    s = await open_fixture_session(own_pool, "after-restart")
    await goto(s, "static.html")


class CrashingBrowserService(LocalBrowserService):
    """Serves fixtures, and kills Chromium right after the first click."""

    crashed = False

    async def open(
        self, session_id: str, *, allowed_domains: Sequence[str], vision: bool = False
    ) -> None:
        await super().open(session_id, allowed_domains=allowed_domains, vision=vision)
        session = self.pool.get(session_id)
        assert session is not None
        await serve_fixtures(session.context)

    async def execute(
        self,
        session_id: str,
        action: AnyAction,
        *,
        expected_version: str | None = None,
        secrets: Mapping[str, Secret] | None = None,
    ) -> ActionResult:
        result = await super().execute(
            session_id, action, expected_version=expected_version, secrets=secrets
        )
        if isinstance(action, Click) and not self.crashed:
            self.crashed = True
            kill_marked(self.pool.marker)
        return result


async def test_run_continues_after_the_browser_dies(own_pool: BrowserPool) -> None:
    settings = make_settings(Path("unused"))
    decide = ScriptedChatModel(
        script=[
            PLAN,
            tool("TypeText", index=1, text="kettle", submit=True),
            tool("Click", index=1),  # "Blue Kettle" → then Chromium is killed
            tool("Done", result="on the detail page"),
        ]
    )
    browser = CrashingBrowserService(own_pool)
    ctx = AgentContext(
        browser=browser,
        llm_decide=decide,
        llm_small=ScriptedChatModel(),
        settings=settings,
        prices=PriceTable.from_settings(settings),
    )
    state = initial_state(
        ctx,
        task="open the blue kettle",
        allowed_domains=["fixtures.test"],
        start_url="https://fixtures.test/search/index.html",
        limits=limits_from(settings),
    )
    final, _ = await drive(
        compile_graph(InMemorySaver()), ctx, state["run_id"], state, max_steps=10
    )
    assert browser.crashed
    assert final["status"] == "done", final["failure_reason"]
    # The last decision saw the restored page: same URL as before the crash.
    assert "/search/detail.html?id=" in final["current_url"]
    assert "URL: https://fixtures.test/search/detail.html?id=" in str(decide.calls[-1][-1].content)
