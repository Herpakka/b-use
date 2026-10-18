"""The browser client the agent talks to. Pure interface: no Playwright import here.

Today the implementation is in-process (``browser.local``); it can become an RPC client later
without the agent changing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol

from browser.actions import AnyAction
from browser.models import ActionResult, BrowserHealth, Observation, SessionInfo
from core.secrets import Secret


class SessionNotFoundError(LookupError):
    """The session does not exist — never opened, closed, or lost with a crashed browser.
    Callers re-open it (cookies and storage come back from disk)."""


class BrowserService(Protocol):
    async def open(
        self, session_id: str, *, allowed_domains: Sequence[str], vision: bool = False
    ) -> None: ...

    async def has_session(self, session_id: str) -> bool: ...

    async def observe(self, session_id: str, *, screenshot: bool = False) -> Observation: ...

    async def execute(
        self,
        session_id: str,
        action: AnyAction,
        *,
        expected_version: str | None = None,
        secrets: Mapping[str, Secret] | None = None,
    ) -> ActionResult:
        """``secrets``: values for the ``{{SECRET:name}}`` placeholders of a TypeText. They are
        typed, never returned, and masked in everything read from the session afterwards."""
        ...

    async def screenshot(self, session_id: str) -> bytes: ...

    async def close(self, session_id: str) -> None:
        """Close the session, saving its cookies/storage for the next run that uses the id."""
        ...

    async def forget(self, session_id: str) -> None:
        """Close the session and delete its saved cookies/storage."""
        ...

    async def list_sessions(self) -> list[SessionInfo]: ...

    async def health(self) -> BrowserHealth: ...

    async def hold(self, session_id: str, held: bool) -> None:
        """Mark a session as waiting for a human (longer idle timeout). Unknown ids: no-op."""
        ...

    def screencast(self, session_id: str) -> AsyncIterator[bytes]:
        """Live JPEG frames; raises SessionNotFoundError when iterated if the id is unknown."""
        ...
