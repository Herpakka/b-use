"""One BrowserSession = one BrowserContext (cookies, storage, tabs) owned by one run at a time."""

from __future__ import annotations

import asyncio
import base64
import time

from playwright.async_api import BrowserContext, Dialog, Page
from playwright.async_api import Error as PlaywrightError

from browser.actions import AnyAction
from browser.dom import ElementRef, build_observation
from browser.executor import execute_action
from browser.guards import NetworkPolicy
from browser.models import ActionResult, Observation, clip_message, short_url
from browser.screencast import take_screenshot
from core.config import Settings


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
        self.refs: dict[int, ElementRef] = {}
        self.dom_version: str | None = None
        self.last_observation: Observation | None = None
        self._page: Page | None = None
        self._attached: set[Page] = set()
        self._dialogs: list[str] = []
        self._blocked: list[str] = []  # reported (then cleared) by the next observation
        self._blocked_total = 0
        self._last_blocked = ""

    async def start(self) -> None:
        self.context.on("page", self._attach)
        page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        self._attach(page)

    def _attach(self, page: Page) -> None:
        """Track a new tab and make it the active one (popups / target=_blank)."""
        if page not in self._attached:
            self._attached.add(page)
            page.on("dialog", self._on_dialog)
            page.set_default_timeout(self.settings.action_timeout_ms)
            page.set_default_navigation_timeout(self.settings.nav_timeout_ms)
        self._page = page

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

    @property
    def blocked_count(self) -> int:
        return self._blocked_total

    @property
    def last_blocked(self) -> str:
        return self._last_blocked

    def touch(self) -> None:
        self.last_used = time.monotonic()

    async def current_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            open_pages = [p for p in self.context.pages if not p.is_closed()]
            self._page = open_pages[-1] if open_pages else await self.context.new_page()
            self._attach(self._page)
        return self._page

    async def observe(self, *, screenshot: bool = False) -> Observation:
        async with self.lock:
            self.touch()
            page = await self.current_page()
            obs, refs = await build_observation(
                page,
                expansion_px=self.settings.viewport_expansion_px,
                max_elements=self.settings.max_elements,
            )
            update: dict[str, object] = {"dialogs": self._dialogs, "blocked": self._blocked}
            if screenshot:
                update["screenshot_b64"] = base64.b64encode(await take_screenshot(page)).decode()
            obs = obs.model_copy(update=update)
            self._dialogs, self._blocked = [], []
            self.refs, self.dom_version, self.last_observation = refs, obs.dom_version, obs
            return obs

    async def execute(
        self, action: AnyAction, *, expected_version: str | None = None
    ) -> ActionResult:
        async with self.lock:
            self.touch()
            return await execute_action(self, action, expected_version=expected_version)

    async def screenshot(self) -> bytes:
        async with self.lock:
            self.touch()
            return await take_screenshot(await self.current_page())
