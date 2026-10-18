"""In-process BrowserService backed by a BrowserPool."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager

from playwright.async_api import Error as PlaywrightError

from browser.actions import AnyAction
from browser.models import ActionResult, BrowserHealth, Observation, SessionInfo
from browser.pool import BrowserPool
from browser.service import SessionNotFoundError
from browser.session import BrowserSession
from core.config import Settings
from core.secrets import Secret

# What Playwright says when the page/context/browser under a call has gone away.
_GONE = ("Target closed", "has been closed", "Browser closed", "Connection closed")


def _gone(err: PlaywrightError) -> bool:
    return any(marker in str(err) for marker in _GONE)


class LocalBrowserService:
    def __init__(self, pool: BrowserPool) -> None:
        self.pool = pool

    def _get(self, session_id: str) -> BrowserSession:
        session = self.pool.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    async def open(
        self, session_id: str, *, allowed_domains: Sequence[str], vision: bool = False
    ) -> None:
        await self.pool.open_session(session_id, allowed_domains=allowed_domains, vision=vision)

    async def has_session(self, session_id: str) -> bool:
        return self.pool.get(session_id) is not None

    async def _lost(self, session: BrowserSession, err: PlaywrightError) -> bool:
        """True (and the session is forgotten) if ``err`` means its browser/context is gone."""
        if session.closed or _gone(err) or not self.pool.connected:
            await self.pool.discard(session.id)
            return True
        return False

    async def observe(self, session_id: str, *, screenshot: bool = False) -> Observation:
        session = self._get(session_id)
        try:
            return await session.observe(screenshot=screenshot)
        except PlaywrightError as err:
            if await self._lost(session, err):
                raise SessionNotFoundError(session_id) from err
            raise

    async def execute(
        self,
        session_id: str,
        action: AnyAction,
        *,
        expected_version: str | None = None,
        secrets: Mapping[str, Secret] | None = None,
    ) -> ActionResult:
        session = self._get(session_id)
        try:
            return await session.execute(action, expected_version=expected_version, secrets=secrets)
        except PlaywrightError as err:
            if await self._lost(session, err):
                raise SessionNotFoundError(session_id) from err
            raise

    async def screenshot(self, session_id: str) -> bytes:
        session = self._get(session_id)
        try:
            return await session.screenshot()
        except PlaywrightError as err:
            if await self._lost(session, err):
                raise SessionNotFoundError(session_id) from err
            raise

    async def close(self, session_id: str) -> None:
        await self.pool.close_session(session_id)

    async def forget(self, session_id: str) -> None:
        await self.pool.forget_session(session_id)

    async def list_sessions(self) -> list[SessionInfo]:
        return [s.info() for s in self.pool.sessions()]

    async def health(self) -> BrowserHealth:
        return self.pool.health()

    async def hold(self, session_id: str, held: bool) -> None:
        if session := self.pool.get(session_id):
            session.held = held
            session.touch()

    async def screencast(self, session_id: str) -> AsyncIterator[bytes]:
        async for frame in self._get(session_id).frames():
            yield frame


@asynccontextmanager
async def local_browser(settings: Settings) -> AsyncIterator[LocalBrowserService]:
    """Start a BrowserPool and expose it as a BrowserService for the run's lifetime."""
    async with BrowserPool(settings) as pool:
        yield LocalBrowserService(pool)
