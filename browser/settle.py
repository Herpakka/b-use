"""Wait for the DOM to stop changing — deliberately *not* ``networkidle``.

Pages with analytics beacons, long-polling or websockets never go network-idle; what the agent
needs is a DOM that holds still long enough to be observed.
"""

from __future__ import annotations

import asyncio
import contextlib

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page

_NODE_COUNT_JS = "() => document.getElementsByTagName('*').length"


async def settle(
    page: Page,
    *,
    interval_s: float = 0.25,
    min_wait_s: float = 0.3,
    timeout_s: float = 5.0,
    stable_polls: int = 2,
) -> bool:
    """Return True once the element count is unchanged for ``stable_polls`` polls in a row.

    Returns False (never raises) if the page is still changing at ``timeout_s``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    with contextlib.suppress(PlaywrightError):
        await page.wait_for_load_state("domcontentloaded", timeout=timeout_s * 1000)
    await asyncio.sleep(min_wait_s)
    last: int | None = None
    stable = 0
    while loop.time() < deadline:
        try:
            count = int(await page.evaluate(_NODE_COUNT_JS))
        except PlaywrightError:
            # Execution context destroyed: a navigation is in flight. Start over.
            last, stable = None, 0
            await asyncio.sleep(interval_s)
            continue
        if count == last:
            stable += 1
            if stable >= stable_polls:
                return True
        else:
            stable = 0
        last = count
        await asyncio.sleep(interval_s)
    return False
