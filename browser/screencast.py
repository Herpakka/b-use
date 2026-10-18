"""Screenshots (debug, vision) and a CDP screencast stream (live view, phase 3 WebSocket)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncGenerator
from typing import Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page


async def take_screenshot(page: Page, *, quality: int = 60) -> bytes:
    return await page.screenshot(type="jpeg", quality=quality)


async def screencast(
    page: Page, *, quality: int = 50, max_width: int = 1280, buffer: int = 2
) -> AsyncGenerator[bytes, None]:
    """Yield JPEG frames as Chromium paints them. Slow consumers drop old frames, never block."""
    cdp = await page.context.new_cdp_session(page)
    frames: asyncio.Queue[bytes] = asyncio.Queue(maxsize=buffer)
    acks: set[asyncio.Task[Any]] = set()

    def on_frame(params: dict[str, Any]) -> None:
        task = asyncio.ensure_future(
            cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
        )
        acks.add(task)
        task.add_done_callback(acks.discard)
        if frames.full():
            frames.get_nowait()
        frames.put_nowait(base64.b64decode(params["data"]))

    cdp.on("Page.screencastFrame", on_frame)
    await cdp.send(
        "Page.startScreencast", {"format": "jpeg", "quality": quality, "maxWidth": max_width}
    )
    try:
        while True:
            yield await frames.get()
    finally:
        with contextlib.suppress(PlaywrightError):
            await cdp.send("Page.stopScreencast")
            await cdp.detach()
