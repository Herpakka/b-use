"""One BrowserSession = one BrowserContext (cookies, storage, tabs) owned by one run at a time."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from collections.abc import AsyncGenerator, Mapping

from playwright.async_api import BrowserContext, Dialog, Page
from playwright.async_api import Error as PlaywrightError

from browser.actions import AnyAction
from browser.dom import ElementRef, build_observation
from browser.executor import execute_action
from browser.guards import NetworkPolicy, navigation_reason
from browser.models import ActionResult, Observation, SessionInfo, clip_message, short_url
from browser.screencast import screencast, take_screenshot
from core.config import Settings
from core.secrets import Secret, SecretRedactor


class BrowserSession:
    def __init__(
        self,
        session_id: str,
        context: BrowserContext,
        policy: NetworkPolicy,
        settings: Settings,
        *,
        vision: bool = False,
    ) -> None:
        self.id = session_id
        self.context = context
        self.policy = policy
        self.settings = settings
        self.vision = vision
        # Serialises every operation: two actions must never interleave on one page.
        self.lock = asyncio.Lock()
        self.last_used = time.monotonic()
        self.closed = False
        # Set while the run waits for a human: the reaper then allows a much longer idle time,
        # so a login window or half-filled form is still there when the answer comes.
        self.held = False
        self.refs: dict[int, ElementRef] = {}
        self.dom_version: str | None = None
        self.last_observation: Observation | None = None
        self._page: Page | None = None
        self._attached: set[Page] = set()
        self._dialogs: list[str] = []
        self._notices: list[str] = []
        self._crashed: set[Page] = set()
        self.saved_at = 0.0  # monotonic time cookies/storage were last written to disk
        self._blocked: list[str] = []  # reported (then cleared) by the next observation
        self._blocked_total = 0  # refused navigations (router or redirect check)
        self._egress_total = 0  # requests the egress proxy refused (any kind)
        self._egress_seen = 0  # … as of the last observation
        self._last_blocked = ""
        self._last_allowed: dict[Page, str] = {}
        self._first_page: Page | None = None
        # Values of secrets typed in this session, masked in everything read back from it.
        self.redactor = SecretRedactor()

    async def start(self) -> None:
        self.context.on("page", self._attach)
        page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        self._first_page = page
        self._attach(page)

    def _attach(self, page: Page) -> None:
        """Track a new tab and make it the active one (popups / target=_blank)."""
        if page not in self._attached:
            self._attached.add(page)
            page.on("dialog", self._on_dialog)
            page.on("crash", self._on_crash)
            page.set_default_timeout(self.settings.action_timeout_ms)
            page.set_default_navigation_timeout(self.settings.nav_timeout_ms)
        self._page = page

    def _on_crash(self, page: Page) -> None:
        self._crashed.add(page)

    async def _on_dialog(self, dialog: Dialog) -> None:
        self._dialogs.append(clip_message(f"{dialog.type}: {dialog.message}"))
        try:
            # Accepting beforeunload lets navigations proceed; everything else is refused.
            if dialog.type == "beforeunload":
                await dialog.accept()
            else:
                await dialog.dismiss()
        except PlaywrightError:
            pass

    def record_blocked(self, url: str, reason: str) -> None:
        self._last_blocked = reason
        self._blocked_total += 1
        self._blocked.append(f"{short_url(url)} ({reason})")

    def record_egress_block(self, target: str, reason: str) -> None:
        self._last_blocked = reason
        self._egress_total += 1
        self._blocked.append(f"{target} ({reason})")

    @property
    def blocked_count(self) -> int:
        return self._blocked_total

    @property
    def egress_block_count(self) -> int:
        return self._egress_total

    def frame_allowed(self, url: str) -> bool:
        if url.startswith(("about:", "data:", "blob:")):
            return True
        return navigation_reason(url, self.policy.allowed_domains) is None

    async def enforce_allowlist(self, *, egress_refused: bool = False) -> str:
        """Undo a navigation that ended outside the allow-list.

        Redirect hops never reach the request router, so a page on an allowed domain can
        bounce the tab anywhere. This runs after every action and before every observation:
        a tab that ended up outside the allowed domains is sent back to its last allowed URL
        (a popup with none is closed). Requests to internal addresses never got this far —
        the egress proxy refused them — but the tab is left on an error page, handled the
        same way. Returns a note for the action's result line ("" if nothing happened).
        """
        page = await self.current_page()
        url = page.url
        if url.startswith("chrome-error://"):
            if not egress_refused:
                return ""  # an ordinary network error: the action's own message says so
            note = f" (navigation blocked: {self._last_blocked})"
        elif url == "about:blank" or self.frame_allowed(url):
            self._last_allowed[page] = url
            return ""
        else:
            reason = navigation_reason(url, self.policy.allowed_domains) or "not allowed"
            self.record_blocked(url, f"redirected outside the allowed domains: {reason}")
            note = f" (redirect to {short_url(url)} blocked)"
        back = self._last_allowed.get(page)
        try:
            if back and back != "about:blank":
                await page.goto(back, wait_until="domcontentloaded")
            elif page is not self._first_page:
                await page.close()  # a popup that never showed an allowed page
            else:
                await page.goto("about:blank")
        except PlaywrightError:
            pass
        return note

    @property
    def last_blocked(self) -> str:
        return self._last_blocked

    def touch(self) -> None:
        self.last_used = time.monotonic()

    async def _reopen_crashed(self, page: Page) -> None:
        """A renderer crash leaves a dead tab ("Aw, Snap"): replace it with a fresh one on the
        same (last allowed) URL."""
        self._crashed.discard(page)
        url = self._last_allowed.get(page) or page.url
        with contextlib.suppress(PlaywrightError):
            await page.close()
        fresh = await self.context.new_page()
        self._attach(fresh)
        if url and url != "about:blank" and self.frame_allowed(url):
            with contextlib.suppress(PlaywrightError):
                await fresh.goto(url, wait_until="domcontentloaded")
        self._notices.append(f"the tab crashed and was reopened at {short_url(url)}")

    async def current_page(self) -> Page:
        if self._page is not None and self._page in self._crashed:
            await self._reopen_crashed(self._page)
        if self._page is None or self._page.is_closed():
            open_pages = [p for p in self.context.pages if not p.is_closed()]
            self._page = open_pages[-1] if open_pages else await self.context.new_page()
            self._attach(self._page)
        return self._page

    async def observe(self, *, screenshot: bool = False) -> Observation:
        async with self.lock:
            self.touch()
            # Pages navigate by themselves too (meta refresh, scripts): a refusal since the last
            # look means an error page here is the proxy's doing, not a network failure.
            await self.enforce_allowlist(egress_refused=self._egress_total > self._egress_seen)
            self._egress_seen = self._egress_total
            page = await self.current_page()
            obs, refs = await build_observation(
                page,
                expansion_px=self.settings.viewport_expansion_px,
                max_elements=self.settings.max_elements,
                frame_allowed=self.frame_allowed,
            )
            update: dict[str, object] = {
                "dialogs": self._dialogs,
                "blocked": self._blocked,
                "notices": self._notices,
            }
            if screenshot:
                update["screenshot_b64"] = base64.b64encode(await take_screenshot(page)).decode()
            obs = obs.model_copy(update=update)
            if self.redactor:
                obs = obs.map_text(self.redactor.scrub)
            self._dialogs, self._blocked, self._notices = [], [], []
            self.refs, self.dom_version, self.last_observation = refs, obs.dom_version, obs
            return obs

    async def execute(
        self,
        action: AnyAction,
        *,
        expected_version: str | None = None,
        secrets: Mapping[str, Secret] | None = None,
    ) -> ActionResult:
        async with self.lock:
            self.touch()
            result = await execute_action(
                self, action, expected_version=expected_version, secrets=secrets
            )
        if not self.redactor:
            return result
        scrub = self.redactor.scrub
        text = result.extracted_text
        return result.model_copy(
            update={
                "message": scrub(result.message),
                "url": scrub(result.url),
                "extracted_text": None if text is None else scrub(text),
            }
        )

    async def screenshot(self) -> bytes:
        async with self.lock:
            self.touch()
            return await take_screenshot(await self.current_page())

    def info(self) -> SessionInfo:
        page = self._page
        return SessionInfo(
            id=self.id,
            url=self.redactor.scrub(page.url) if page is not None and not page.is_closed() else "",
            tabs=len(self.context.pages),
            idle_s=round(time.monotonic() - self.last_used, 1),
            busy=self.lock.locked(),
            held=self.held,
        )

    async def frames(self) -> AsyncGenerator[bytes, None]:
        """Live JPEG frames of whichever tab is active; follows the agent into new tabs.

        Needs no lock: the screencast is a separate CDP channel and never touches the DOM.
        """
        while not self.closed:
            page = await self.current_page()
            stream = screencast(page)
            try:
                async for frame in stream:
                    yield frame
                    if self._page is not page:
                        break
                else:
                    return  # page closed and the stream ended
            finally:
                await stream.aclose()
