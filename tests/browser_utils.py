"""Serve tests/fixtures/ from fake hosts via request interception — no local HTTP server.

The fixture route is registered after the network guard, so Playwright runs it first; the guard
(and its SSRF blocking) stays fully active for every other host.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Route

from browser.actions import Navigate
from browser.models import Observation
from browser.session import BrowserSession

FIXTURES = (Path(__file__).parent / "fixtures").resolve()
FIXTURE_HOSTS = re.compile(r"^https://(fixtures|other)\.test/")
FIXTURE_DOMAINS = ["fixtures.test", "other.test"]


async def serve_fixtures(context: BrowserContext) -> None:
    async def handler(route: Route) -> None:
        rel = urlsplit(route.request.url).path.lstrip("/") or "static.html"
        path = (FIXTURES / rel).resolve()
        if FIXTURES not in path.parents or not path.is_file():
            await route.fulfill(status=404, body="not found", content_type="text/plain")
            return
        await route.fulfill(path=str(path))

    await context.route(FIXTURE_HOSTS, handler)


async def goto(session: BrowserSession, path: str) -> Observation:
    result = await session.execute(Navigate(url=f"https://fixtures.test/{path.lstrip('/')}"))
    assert result.ok, result.message
    return await session.observe()


def find(obs: Observation, name: str) -> int:
    """Index of the first element whose name contains ``name``."""
    for e in obs.elements:
        if name in e.name:
            return e.index
    raise AssertionError(f"no element named {name!r} in:\n{obs.render()}")
