"""Chromium is launched once; every session gets its own BrowserContext.

Sessions persist cookies/localStorage across runs via ``storage_state`` JSON files keyed by
session id. A reaper closes sessions idle too long.

Health: a probe asks the browser for its version every ``browser_probe_interval_s``. A browser
that crashed (disconnected) is relaunched at once; one that stops answering twice in a row is
killed and relaunched. Either way its sessions are dropped, and the agent restores them from
the cookies saved here (at most one probe interval old) and its last URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import secrets
import time
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from urllib.parse import urlsplit

import psutil
from playwright.async_api import Browser, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError

from browser.egress import EgressProxy, ProxySettings, context_proxy
from browser.guards import NetworkPolicy, install_routes, normalize_domains
from browser.models import BrowserHealth
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


REMOTE_EGRESS_URL = "http://127.0.0.1:3129"


def endpoint_token(endpoint: str) -> str:
    return urlsplit(endpoint).path.strip("/")


def redact_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}/…"


def kill_marked(marker: str) -> int:
    """SIGKILL the processes launched with ``marker`` on their command line. A frozen browser
    never answers ``close()``, and stopping Playwright then hangs too."""
    killed = 0
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if marker in (proc.info["cmdline"] or []):
                proc.kill()
                killed += 1
        except psutil.Error:
            continue
    return killed


class BrowserPool:
    def __init__(self, settings: Settings, *, egress: EgressProxy | None = None) -> None:
        self.settings = settings
        # A remote browser's traffic leaves from *its* machine, so its egress proxy runs there
        # (devtools.browser_host); only a local browser gets one in this process.
        if egress is None and settings.egress_proxy and settings.browser_ws_endpoint is None:
            egress = EgressProxy(host=settings.egress_listen_host, port=settings.egress_listen_port)
        self.egress = egress
        if egress is not None and egress.on_blocked is None:
            egress.on_blocked = self._on_egress_blocked
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._sessions: dict[str, BrowserSession] = {}
        self._slots = asyncio.Semaphore(settings.max_contexts)
        self._lock = asyncio.Lock()
        self._reaper: asyncio.Task[None] | None = None
        self._prober: asyncio.Task[None] | None = None
        # An unknown switch Chromium ignores; it lets us find (and kill) exactly our processes.
        self.marker = f"--bu-instance={secrets.token_hex(6)}"
        self.restarts = 0
        self._health = "not started"
        self._probe_failures = 0

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
        if self.egress is not None:
            await self.egress.start()
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
        self._prober = asyncio.create_task(self._probe_loop(), name="browser-probe")

    async def stop(self) -> None:
        for task in (self._reaper, self._prober):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        for session_id in list(self._sessions):
            await self.close_session(session_id)
        browser, self._browser = self._browser, None  # a deliberate close, not a crash
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._pw = None
        if self.egress is not None:
            await self.egress.stop()

    async def _launch(self) -> Browser:
        assert self._pw is not None, "BrowserPool.start() was not called"
        if endpoint := self.settings.browser_ws_endpoint:
            # A browser someone else runs (e.g. a visible Chromium on the developer's desktop,
            # see devtools/browser_host.py). Contexts, routes and CDP all work over the link.
            browser = await self._pw.chromium.connect(
                endpoint, timeout=15_000, headers=loopback_host_header(endpoint)
            )
            log.info(
                "chromium_connected", version=browser.version, endpoint=redact_endpoint(endpoint)
            )
        else:
            browser = await self._pw.chromium.launch(
                headless=self.settings.headless,
                args=["--disable-dev-shm-usage", self.marker],
            )
            log.info("chromium_launched", version=browser.version)
        browser.on("disconnected", self._on_disconnected)
        self._browser = browser
        self._health, self._probe_failures = "ok", 0
        return browser

    def _on_disconnected(self, browser: Browser) -> None:
        if browser is not self._browser:
            return  # one we already replaced
        log.warning("chromium_disconnected", sessions_lost=len(self._sessions))
        self._health = "not connected"
        self._drop_sessions()

    def _drop_sessions(self) -> None:
        """Forget every session: their contexts died with the browser."""
        for session in self._sessions.values():
            session.closed = True
            self._slots.release()
        self._sessions.clear()

    @property
    def connected(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    def health(self) -> BrowserHealth:
        ok = self.connected and self._health == "ok"
        return BrowserHealth(
            ok=ok,
            detail="ok" if ok else (self._health if self._health != "ok" else "not connected"),
            version=self._browser.version if self.connected and self._browser else None,
            remote=self.settings.browser_ws_endpoint is not None,
            sessions=len(self._sessions),
            max_sessions=self.settings.max_contexts,
            restarts=self.restarts,
        )

    async def probe(self) -> str | None:
        """None if the browser answers in time, else why not."""
        browser = self._browser
        if browser is None or not browser.is_connected():
            return "not connected"
        timeout = self.settings.browser_probe_timeout_s
        try:
            cdp = await asyncio.wait_for(browser.new_browser_cdp_session(), timeout)
            try:
                await asyncio.wait_for(cdp.send("Browser.getVersion"), timeout)
            finally:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(cdp.detach(), 2)
        except TimeoutError:
            return f"no answer within {timeout:g}s"
        except PlaywrightError as err:
            return (str(err).strip().splitlines() or ["error"])[0][:120]
        return None

    async def restart(self, reason: str) -> None:
        """Replace the browser (killing ours if it is frozen) and drop its sessions."""
        async with self._lock:
            old, self._browser = self._browser, None
            self.restarts += 1
            log.error("chromium_restart", reason=reason, sessions_lost=len(self._sessions))
            self._drop_sessions()
            if old is not None:
                if self.settings.browser_ws_endpoint is None:
                    kill_marked(self.marker)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(old.close(), 5)
            await self._launch()

    async def probe_once(self) -> None:
        reason = await self.probe()
        if reason is None:
            self._health, self._probe_failures = "ok", 0
            await self.checkpoint_storage()
            return
        self._probe_failures += 1
        previous, self._health = self._health, reason
        if reason == "not connected":
            async with self._lock:
                if not self.connected:  # crashed, or a remote browser that isn't up (yet)
                    self._drop_sessions()
                    try:
                        await self._launch()
                    except Exception as err:
                        if previous != reason:
                            log.warning(
                                "chromium_unavailable", error=str(err).splitlines()[0][:200]
                            )
        elif self._probe_failures >= 2:
            await self.restart(reason)

    async def checkpoint_storage(self) -> None:
        """Save cookies/storage of sessions used since their last save, so a browser crash
        loses at most one probe interval of logins."""
        for session in list(self._sessions.values()):
            if session.closed or session.lock.locked() or session.last_used <= session.saved_at:
                continue
            try:
                async with session.lock:
                    await self._persist(session)
            except Exception as err:
                log.warning(
                    "storage_checkpoint_failed", session_id=session.id, error=str(err)[:200]
                )

    async def _probe_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.browser_probe_interval_s)
            try:
                await self.probe_once()
            except Exception as err:  # the probe must never die
                log.error("browser_probe_failed", error=str(err)[:200])

    async def _ensure_browser(self) -> Browser:
        if self._browser is None or not self._browser.is_connected():
            if self._browser is not None:
                log.warning("chromium_relaunch", sessions_lost=len(self._sessions))
            self._drop_sessions()
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

    def _on_egress_blocked(self, session_id: str | None, target: str, reason: str) -> None:
        session = self._sessions.get(session_id) if session_id else None
        if session is not None:
            session.record_egress_block(target, reason)

    def context_proxy(self, session_id: str) -> ProxySettings | None:
        """Proxy settings for a new context; None only if the egress proxy is switched off."""
        if self.egress is not None:
            url = self.settings.egress_browser_url or f"http://127.0.0.1:{self.egress.port}"
            return self.egress.context_proxy(session_id, url)
        if self.settings.egress_proxy and (endpoint := self.settings.browser_ws_endpoint):
            url = self.settings.egress_browser_url or REMOTE_EGRESS_URL
            return context_proxy(session_id, url, endpoint_token(endpoint))
        return None

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
            proxy=self.context_proxy(session_id),  # type: ignore[arg-type]
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

    async def discard(self, session_id: str) -> None:
        """Forget a session whose context is already gone (nothing left to save)."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.closed = True
        self._slots.release()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(session.context.close(), 5)
        log.warning("session_lost", session_id=session_id)

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

    async def forget_session(self, session_id: str) -> None:
        if not _SESSION_ID.match(session_id):
            raise ValueError(f"invalid session id {session_id!r}")
        await self.close_session(session_id, persist=False)
        self.storage_path(session_id).unlink(missing_ok=True)

    async def _persist(self, session: BrowserSession) -> None:
        path = self.storage_path(session.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        await session.context.storage_state(path=str(path))
        session.saved_at = time.monotonic()
        # Holds login cookies: owner-only.
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)

    async def reap_idle(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now

        def limit(s: BrowserSession) -> float:
            return self.settings.held_idle_timeout_s if s.held else self.settings.idle_timeout_s

        idle = [
            sid
            for sid, s in self._sessions.items()
            if not s.lock.locked() and now - s.last_used > limit(s)
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
