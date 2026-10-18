import asyncio
import base64

import pytest

from browser.screencast import screencast
from browser.session import BrowserSession
from tests.browser_utils import goto

pytestmark = pytest.mark.browser

JPEG_MAGIC = b"\xff\xd8\xff"


async def test_screenshot_is_jpeg(session: BrowserSession) -> None:
    await goto(session, "static.html")
    assert (await session.screenshot()).startswith(JPEG_MAGIC)
    obs = await session.observe(screenshot=True)
    assert obs.screenshot_b64 and base64.b64decode(obs.screenshot_b64).startswith(JPEG_MAGIC)


async def test_screencast_yields_frames(session: BrowserSession) -> None:
    await goto(session, "infinite_scroll.html")
    page = await session.current_page()
    stream = screencast(page)

    async def first_frame() -> bytes:
        async for frame in stream:
            return frame
        raise AssertionError("stream ended")

    task = asyncio.create_task(first_frame())
    for _ in range(10):  # make Chromium paint
        await page.mouse.wheel(0, 200)
        await asyncio.sleep(0.1)
    frame = await asyncio.wait_for(task, timeout=5)
    await stream.aclose()
    assert frame.startswith(JPEG_MAGIC)
