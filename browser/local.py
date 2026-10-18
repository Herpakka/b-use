"""In-process BrowserService backed by a BrowserPool."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from browser.actions import AnyAction
from browser.models import ActionResult, Observation, SessionInfo
from browser.pool import BrowserPool
from browser.service import SessionNotFoundError
from browser.session import BrowserSession
from core.config import Settings


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

    async def observe(self, session_id: str, *, screenshot: bool = False) -> Observation:
        return await self._get(session_id).observe(screenshot=screenshot)

    async def execute(
        self, session_id: str, action: AnyAction, *, expected_version: str | None = None
    ) -> ActionResult:
        return await self._get(session_id).execute(action, expected_version=expected_version)

    async def screenshot(self, session_id: str) -> bytes:
        return await self._get(session_id).screenshot()

    async def close(self, session_id: str) -> None:
        await self.pool.close_session(session_id)

    async def list_sessions(self) -> list[SessionInfo]:
        return [s.info() for s in self.pool.sessions()]

    async def screencast(self, session_id: str) -> AsyncIterator[bytes]:
        async for frame in self._get(session_id).frames():
            yield frame


@asynccontextmanager
async def local_browser(settings: Settings) -> AsyncIterator[LocalBrowserService]:
    """Start a BrowserPool and expose it as a BrowserService for the run's lifetime."""
    async with BrowserPool(settings) as pool:
        yield LocalBrowserService(pool)
