"""Chromium is launched once; every session gets its own BrowserContext.

Sessions persist cookies/localStorage across runs via ``storage_state`` JSON files keyed by
session id (phase 3 moves these into Postgres). A reaper closes sessions idle too long.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from urllib.parse import urlsplit

from playwright.async_api import Browser, Playwright, async_playwright

from browser.guards import NetworkPolicy, install_routes, normalize_domains
from browser.session import BrowserSession
from core.config import Settings
from core.logging import get_logger

log = get_logger("browser.pool")

_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class PoolExhaustedError(RuntimeError):
    pass


def loopback_host_header(endpoint: str) -> dict[str, str]:
    """Playwright's browser server, when bound to loopback, only accepts loopback Host headers
    (DNS-rebinding guard). Docker Desktop forwards host.docker.internal to the host's loopback,
    so present ourselves as localhost."""
    port = urlsplit(endpoint).port
    return {"Host": f"localhost:{port}"} if port else {}


def redact_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}/…"


class BrowserPool:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._sessions: dict[str, BrowserSession] = {}
        self._slots = asyncio.Semaphore(settings.max_contexts)
        self._lock = asyncio.Lock()
        self._reaper: asyncio.Task[None] | None = None

    async def __aenter__(self) -> BrowserPool:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        self._pw = await async_playwright().start()
        try:
            await self._launch()
        except Exception as err:
            if not self.settings.browser_ws_endpoint:
                raise
            # A remote browser may simply not be up yet; connect on the first session instead
            # of refusing to start the whole service.
            log.warning("remote_browser_unavailable", error=str(err).splitlines()[0][:200])
        self._reaper = asyncio.create_task(self._reap_loop(), name="browser-reaper")

    async def stop(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
        for session_id in list(self._sessions):
            await self.close_session(session_id)
        if self._browser is not None:
            with contextlib.suppress(Exception):
                await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._browser = self._pw = None

    async def _launch(self) -> Browser:
        assert self._pw is not None, "BrowserPool.start() was not called"
        if endpoint := self.settings.browser_ws_endpoint:
            # A browser someone else runs (e.g. a visible Chromium on the developer's desktop,
            # see devtools/browser_host.py). Contexts, routes and CDP all work over the link.
            self._browser = await self._pw.chromium.connect(
                endpoint, timeout=15_000, headers=loopback_host_header(endpoint)
            )
            log.info(
                "chromium_connected",
                version=self._browser.version,
                endpoint=redact_endpoint(endpoint),
            )
            return self._browser
        self._browser = await self._pw.chromium.launch(
            headless=self.settings.headless,
            args=["--disable-dev-shm-usage"],
        )
        log.info("chromium_launched", version=self._browser.version)
        return self._browser

    async def _ensure_browser(self) -> Browser:
        if self._browser is None or not self._browser.is_connected():
            if self._browser is not None:
                log.warning("chromium_relaunch", sessions_lost=len(self._sessions))
            for session in self._sessions.values():
                session.closed = True
                self._slots.release()
            self._sessions.clear()
            return await self._launch()
        return self._browser

    def get(self, session_id: str) -> BrowserSession | None:
        session = self._sessions.get(session_id)
        return session if session is not None and not session.closed else None

    @property
    def active(self) -> int:
        return len(self._sessions)

    def sessions(self) -> list[BrowserSession]:
        return [s for s in self._sessions.values() if not s.closed]

    def storage_path(self, session_id: str) -> Path:
        return self.settings.sessions_dir / f"{session_id}.json"

    async def open_session(
        self,
        session_id: str,
        *,
        allowed_domains: Iterable[str],
        vision: bool = False,
        resolve_dns: bool = True,
    ) -> BrowserSession:
        if not _SESSION_ID.match(session_id):
            raise ValueError(f"invalid session id {session_id!r}")
        if existing := self.get(session_id):
            return existing
        try:
            await asyncio.wait_for(
                self._slots.acquire(), timeout=self.settings.pool_acquire_timeout_s
            )
        except TimeoutError as err:
            raise PoolExhaustedError(
                f"all {self.settings.max_contexts} browser contexts are busy"
            ) from err
        async with self._lock:
            if existing := self.get(session_id):
                self._slots.release()
                return existing
            try:
                session = await self._create(session_id, allowed_domains, vision, resolve_dns)
            except BaseException:
                self._slots.release()
                raise
            self._sessions[session_id] = session
        log.info("session_opened", session_id=session_id, domains=session.policy.allowed_domains)
        return session

    async def _create(
        self, session_id: str, allowed_domains: Iterable[str], vision: bool, resolve_dns: bool
    ) -> BrowserSession:
        browser = await self._ensure_browser()
        storage = self.storage_path(session_id)
        context = await browser.new_context(
            viewport={
                "width": self.settings.viewport_width,
                "height": self.settings.viewport_height,
            },
            storage_state=str(storage) if storage.exists() else None,
            # Service-worker requests bypass context.route, so the network guard needs them off.
            service_workers="block",
            accept_downloads=False,
        )
        policy = NetworkPolicy(
            allowed_domains=normalize_domains(allowed_domains),
            block_resources=self.settings.block_resources and not vision,
            resolve_dns=resolve_dns,
        )
        session = BrowserSession(session_id, context, policy, self.settings, vision=vision)
        policy.on_blocked = session.record_blocked
        try:
            await install_routes(context, policy)
            await session.start()
        except BaseException:
            await context.close()
            raise
        return session

    async def close_session(self, session_id: str, *, persist: bool = True) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        try:
            async with session.lock:
                if persist:
                    await self._persist(session)
                await session.context.close()
        except Exception as err:
            log.warning("session_close_failed", session_id=session_id, error=str(err)[:200])
        finally:
            session.closed = True
            self._slots.release()
            log.info("session_closed", session_id=session_id)

    async def _persist(self, session: BrowserSession) -> None:
        path = self.storage_path(session.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        await session.context.storage_state(path=str(path))
        # Holds login cookies: owner-only.
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)

    async def reap_idle(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        idle = [
            sid
            for sid, s in self._sessions.items()
            if not s.lock.locked() and now - s.last_used > self.settings.idle_timeout_s
        ]
        for sid in idle:
            log.info("session_reaped", session_id=sid)
            await self.close_session(sid)
        return idle

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.reaper_interval_s)
            try:
                await self.reap_idle()
            except Exception as err:  # the reaper must never die
                log.error("reaper_failed", error=str(err)[:200])
