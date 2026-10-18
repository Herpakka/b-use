"""Map an action onto Playwright and report back in one short line.

Messages double as the agent's step history (``clicked [2] "Add to cart" → ok``), so they stay
short and never include secrets.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Locator, Page

from browser.actions import AnyAction, Click, Done, Extract, Navigate, Scroll, TypeText
from browser.dom import FINGERPRINT_JS, ElementRef, page_text
from browser.guards import navigation_reason
from browser.models import ActionResult, clip_message, short_url
from browser.settle import settle

if TYPE_CHECKING:
    from browser.session import BrowserSession


def explain(err: PlaywrightError) -> str:
    text = str(err)
    first = text.strip().splitlines()[0] if text.strip() else type(err).__name__
    if "intercepts pointer events" in text:
        return "another element covers it (overlay or modal?)"
    if "ERR_BLOCKED_BY_CLIENT" in text:
        return "blocked by the network policy"
    if "ERR_TUNNEL_CONNECTION_FAILED" in text or "ERR_PROXY_CONNECTION_FAILED" in text:
        return "refused by the egress proxy (internal address), or the proxy is not running"
    if "ERR_NAME_NOT_RESOLVED" in text:
        return "domain does not resolve"
    if "not an <input>" in text or "not an input" in text.lower():
        return "element is not a text field"
    if "Timeout" in first:
        return "timed out"
    return first[:120]


def _quote(text: str, limit: int = 40) -> str:
    text = " ".join(text.split())
    return '"' + (text if len(text) <= limit else text[: limit - 1] + "…") + '"'


def describe(action: AnyAction, ref: ElementRef | None = None) -> str:
    target = ""
    if isinstance(action, Click | TypeText):
        target = f"[{action.index}]" + (f" {_quote(ref.name)}" if ref and ref.name else "")
    match action:
        case Navigate():
            return f"navigate {short_url(action.url)}"
        case Click():
            return f"click {target}"
        case TypeText():
            shown = "••••" if ref is not None and ref.type == "password" else _quote(action.text)
            return f"type {shown} into {target}" + (" + Enter" if action.submit else "")
        case Scroll():
            return f"scroll {action.direction} {action.pages:g} page(s)"
        case Extract():
            return f"extract {_quote(action.query, 60)}"
        case Done():
            return "done"


def _result(ok: bool, message: str, page: Page, **extra: object) -> ActionResult:
    return ActionResult.model_validate(
        {"ok": ok, "message": clip_message(message), "url": page.url, **extra}
    )


async def _resolve(
    session: BrowserSession, index: int, expected_version: str | None
) -> ElementRef | str:
    """Return the element ref, or ``"stale"`` / ``"missing"``."""
    if expected_version is not None and expected_version != session.dom_version:
        return "stale"
    ref = session.refs.get(index)
    if ref is None:
        return "missing"
    if ref.frame.is_detached():
        return "stale"
    loc = ref.frame.locator(ref.selector)
    try:
        if await loc.count() != 1:
            return "stale"
        fp = await loc.evaluate(FINGERPRINT_JS, timeout=1000)
    except PlaywrightError:
        return "stale"
    return ref if fp == ref.fingerprint else "stale"


async def _after_interaction(
    session: BrowserSession, page: Page, pages_before: int, blocked_before: int, egress_before: int
) -> str:
    await settle(page)
    notes: list[str] = []
    router_blocked = session.blocked_count > blocked_before
    if len(session.context.pages) > pages_before:
        new_page = await session.current_page()
        await settle(new_page)
        if router_blocked and new_page.url == "about:blank":
            await new_page.close()  # a popup whose navigation the guard refused
        else:
            notes.append("opened a new tab")
    redirect_note = await session.enforce_allowlist(
        egress_refused=session.egress_block_count > egress_before
    )
    if router_blocked and not redirect_note:
        notes.append("navigation blocked: domain not allowed")
    return "".join(f" ({n})" for n in notes) + redirect_note


async def execute_action(
    session: BrowserSession, action: AnyAction, *, expected_version: str | None = None
) -> ActionResult:
    page = await session.current_page()
    settings = session.settings
    ref: ElementRef | None = None

    if isinstance(action, Click | TypeText):
        resolved = await _resolve(session, action.index, expected_version)
        if resolved == "stale":
            return _result(
                False,
                f"[{action.index}] changed since the last observation — re-observing",
                page,
                stale=True,
            )
        if resolved == "missing":
            return _result(
                False,
                f"{describe(action)} → error: no element [{action.index}] in the current"
                " observation (use a listed number; scroll if the target is off-screen)",
                page,
            )
        assert isinstance(resolved, ElementRef)
        ref = resolved

    label = describe(action, ref)
    try:
        match action:
            case Navigate():
                if reason := navigation_reason(action.url, session.policy.allowed_domains):
                    return _result(False, f"{label} → blocked: {reason}", page)
                blocked_before = session.blocked_count
                egress_before = session.egress_block_count
                error: PlaywrightError | None = None
                try:
                    await page.goto(action.url, wait_until="domcontentloaded")
                except PlaywrightError as err:
                    error = err
                # A redirect may have left the allow-list or hit an internal address.
                if note := await session.enforce_allowlist(
                    egress_refused=session.egress_block_count > egress_before
                ):
                    return _result(False, f"{label} → blocked{note}", await session.current_page())
                if session.blocked_count > blocked_before:  # e.g. DNS points inside
                    return _result(False, f"{label} → blocked: {session.last_blocked}", page)
                if error is not None:
                    raise error
                await settle(page)
                return _result(True, f"{label} → ok", page)

            case Click():
                assert ref is not None
                if ref.tag == "select":
                    return _result(
                        False, f"{label} → use type_text with an option label for <select>", page
                    )
                pages_before, blocked_before = len(session.context.pages), session.blocked_count
                egress_before = session.egress_block_count
                await _locator(ref).click(timeout=settings.action_timeout_ms)
                note = await _after_interaction(
                    session, page, pages_before, blocked_before, egress_before
                )
                return _result(True, f"{label} → ok{note}", await session.current_page())

            case TypeText():
                assert ref is not None
                loc = _locator(ref)
                if ref.tag == "select":
                    try:
                        await loc.select_option(
                            label=action.text, timeout=settings.action_timeout_ms
                        )
                    except PlaywrightError:
                        await loc.select_option(
                            value=action.text, timeout=settings.action_timeout_ms
                        )
                else:
                    await loc.fill(action.text, timeout=settings.action_timeout_ms)
                note = ""
                if ref.tag == "select":
                    selected = await loc.evaluate(
                        "el => el.selectedOptions.length"
                        " ? el.selectedOptions[0].textContent.trim() : ''"
                    )
                    note = f" (selected {_quote(str(selected))})"
                if action.submit:
                    pages_before, blocked_before = len(session.context.pages), session.blocked_count
                    egress_before = session.egress_block_count
                    await loc.press("Enter", timeout=settings.action_timeout_ms)
                    note += await _after_interaction(
                        session, page, pages_before, blocked_before, egress_before
                    )
                return _result(True, f"{label} → ok{note}", await session.current_page())

            case Scroll():
                viewport = page.viewport_size or {"width": 1280, "height": 800}
                await page.mouse.move(viewport["width"] / 2, viewport["height"] / 2)
                sign = 1 if action.direction == "down" else -1
                await page.mouse.wheel(0, sign * viewport["height"] * action.pages)
                await asyncio.sleep(0.2)
                await settle(page, min_wait_s=0.1, timeout_s=2.0)
                y, height, vh = await page.evaluate(
                    "() => [Math.round(scrollY),"
                    " Math.round((document.scrollingElement || document.documentElement)"
                    ".scrollHeight), innerHeight]"
                )
                below = max(0, int(height) - int(y) - int(vh))
                return _result(True, f"{label} → {y}px above, {below}px below", page)

            case Extract():
                text = await page_text(page, settings.extract_max_chars)
                return _result(True, f"{label} → read {len(text)} chars", page, extracted_text=text)

            case Done():
                return _result(True, "done", page)
    except PlaywrightError as err:
        return _result(False, f"{label} → error: {explain(err)}", page)


def _locator(ref: ElementRef) -> Locator:
    return ref.frame.locator(ref.selector)
