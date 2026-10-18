"""The browser client the agent talks to. Pure interface: no Playwright import here.

Today the implementation is in-process (``browser.local``); it can become an RPC client later
without the agent changing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from browser.actions import AnyAction
from browser.models import ActionResult, Observation, SessionInfo


class SessionNotFoundError(LookupError):
    pass


class BrowserService(Protocol):
    async def open(
        self, session_id: str, *, allowed_domains: Sequence[str], vision: bool = False
    ) -> None: ...

    async def has_session(self, session_id: str) -> bool: ...

    async def observe(self, session_id: str, *, screenshot: bool = False) -> Observation: ...

    async def execute(
        self, session_id: str, action: AnyAction, *, expected_version: str | None = None
    ) -> ActionResult: ...

    async def screenshot(self, session_id: str) -> bytes: ...

    async def close(self, session_id: str) -> None: ...

    async def list_sessions(self) -> list[SessionInfo]: ...

    async def hold(self, session_id: str, held: bool) -> None:
        """Mark a session as waiting for a human (longer idle timeout). Unknown ids: no-op."""
        ...

    def screencast(self, session_id: str) -> AsyncIterator[bytes]:
        """Live JPEG frames; raises SessionNotFoundError when iterated if the id is unknown."""
        ...
