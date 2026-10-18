"""Remote-browser mode: the pool drives a browser served by devtools.browser_host."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import secrets
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from playwright._impl._driver import get_driver_env

from browser.actions import Navigate
from browser.egress import EgressProxy, IPAddress, system_resolve
from browser.pool import BrowserPool, loopback_host_header, redact_endpoint
from devtools.browser_host import server_command
from tests.browser_utils import FIXTURE_DOMAINS, find, goto, serve_fixtures
from tests.conftest import make_settings

pytestmark = pytest.mark.browser


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
async def endpoint() -> AsyncIterator[str]:
    port, token = free_port(), secrets.token_urlsafe(16)
    cmd = server_command(chrome=None, host="127.0.0.1", port=port, token=token, headless=True)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, **get_driver_env()},
    )
    assert proc.stdout is not None
    try:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
    except TimeoutError:
        proc.kill()
        pytest.skip("browser server did not start")
    if not line.startswith(b"ready "):
        proc.kill()
        pytest.skip(f"browser server failed: {line.decode()[:200]}")
    yield f"ws://127.0.0.1:{port}/{token}"
    proc.terminate()
    await proc.wait()


def test_remote_mode_uses_the_proxy_beside_the_browser(tmp_path: Path) -> None:
    remote = make_settings(tmp_path, browser_ws_endpoint="ws://host.docker.internal:3010/tok3n")
    pool = BrowserPool(remote)
    assert pool.egress is None  # no proxy in this process: traffic leaves from the host
    proxy = pool.context_proxy("acme.run-1")
    assert proxy is not None
    assert proxy["server"] == "http://127.0.0.1:3129" and proxy["password"] == "tok3n"
    local = BrowserPool(make_settings(tmp_path))
    assert local.egress is not None


def test_endpoint_helpers() -> None:
    url = "ws://host.docker.internal:3010/s3cret-token"
    assert loopback_host_header(url) == {"Host": "localhost:3010"}
    assert redact_endpoint(url) == "ws://host.docker.internal:3010/…"
    assert "s3cret" not in redact_endpoint(url)


async def test_pool_drives_a_served_browser(endpoint: str, tmp_path: Path) -> None:
    # What browser_host runs beside the browser: the egress proxy, keyed by the endpoint token.
    async def resolve(host: str, port: int) -> list[IPAddress]:
        if host == "internal.test":  # an intranet name that resolves to a private address
            return [ipaddress.ip_address("10.0.0.5")]
        return await system_resolve(host, port)

    egress = EgressProxy(password=endpoint.rsplit("/", 1)[1], resolve=resolve)
    await egress.start()
    pool = BrowserPool(
        make_settings(
            tmp_path,
            browser_ws_endpoint=endpoint,
            egress_browser_url=f"http://127.0.0.1:{egress.port}",
        )
    )
    await pool.start()
    try:
        s = await pool.open_session("remote-1", allowed_domains=FIXTURE_DOMAINS)
        await serve_fixtures(s.context)
        obs = await goto(s, "search/index.html")
        assert find(obs, "Search")
        # the network guard still applies: routes are enforced from this side of the link
        blocked = await s.execute(Navigate(url="http://169.254.169.254/"))
        assert not blocked.ok and "private or reserved" in blocked.message

        frames = s.frames()
        frame = await asyncio.wait_for(anext(frames), timeout=10)
        await frames.aclose()
        assert frame.startswith(b"\xff\xd8\xff")
        assert [i.id for i in [s.info()]] == ["remote-1"]
        # a page request to an internal name passes the router (no DNS there) and is refused
        # by the proxy beside the browser
        page = await s.current_page()
        await page.evaluate(
            "() => fetch('https://internal.test/', {mode: 'no-cors'}).catch(() => null)"
        )
        assert any("internal.test" in target for _, target, _ in egress.blocked)
    finally:
        await pool.stop()
        await egress.stop()
