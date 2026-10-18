"""Redirects and internal targets, through real Chromium, real HTTP redirects and the proxy.

Two local servers stand in for the outside world:
* ``site``     — a "public" site; ``site.test`` (allowed) and ``outside.test`` (not allowed)
                 both resolve to it, and its ``/to?url=…`` answers with a real 302.
* ``internal`` — an internal service (``internal.test``) that must never receive a request;
                 it counts hits so the test can prove nothing got through.
The proxy gets a test resolver for these names; everything else uses the real policy.
"""

from __future__ import annotations

import ipaddress
import secrets
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import pytest

from browser.actions import Click, Extract, Navigate
from browser.egress import EgressProxy, IPAddress, default_block
from browser.pool import BrowserPool
from browser.session import BrowserSession
from tests.browser_utils import find
from tests.conftest import make_settings

pytestmark = pytest.mark.browser

LOOPBACK = ipaddress.ip_address("127.0.0.1")


@dataclass
class Servers:
    site_port: int
    internal_port: int
    internal_hits: list[str]

    def site(self, path: str, host: str = "site.test") -> str:
        return f"http://{host}:{self.site_port}{path}"

    def internal(self, path: str) -> str:
        return f"http://internal.test:{self.internal_port}{path}"

    def bounce(self, target: str) -> str:
        return self.site("/to?url=" + quote(target, safe=""))


def _site_handler(servers: Servers) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def _html(self, body: str) -> None:
            data = f"<!doctype html><html><body>{body}</body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path == "/to":
                self.send_response(302)
                self.send_header("Location", parse_qs(url.query)["url"][0])
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif url.path == "/home":
                s = servers
                self._html(
                    "<title>Home</title><h1>Home page</h1>"
                    f'<a href="{s.bounce(s.site("/evil", "outside.test"))}">Go outside</a>'
                    f'<a href="{s.bounce(s.internal("/secret"))}">Go internal</a>'
                    f'<a target="_blank" href="{s.bounce(s.site("/evil", "outside.test"))}">'
                    "Popup outside</a>"
                )
            elif url.path == "/framed":
                s = servers
                self._html(
                    "<p>Has a frame</p><button>Own button</button>"
                    f'<iframe src="{s.bounce(s.site("/frame-body", "outside.test"))}"'
                    ' width="300" height="120"></iframe>'
                )
            elif url.path == "/subresources":
                s = servers
                self._html(
                    "<p>Loads internal things</p>"
                    f'<img src="{s.internal("/pixel.png")}" width="10" height="10">'
                    f'<img src="{s.bounce(s.internal("/via-redirect.png"))}" width="10">'
                    f"<script>fetch('{s.internal('/fetch')}', {{mode: 'no-cors'}})"
                    ".catch(() => {});</script>"
                )
            elif url.path == "/evil":
                self._html("<title>Evil</title><button>Evil button</button>")
            elif url.path == "/frame-body":
                self._html("<button>Framed evil</button>")
            else:
                self.send_error(404)

    return Handler


@pytest.fixture(scope="module")
def servers() -> Iterator[Servers]:
    hits: list[str] = []

    class Internal(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "6")
            self.end_headers()
            self.wfile.write(b"secret")

    internal = ThreadingHTTPServer(("127.0.0.1", 0), Internal)
    s = Servers(site_port=0, internal_port=internal.server_address[1], internal_hits=hits)
    site = ThreadingHTTPServer(("127.0.0.1", 0), _site_handler(s))
    s.site_port = site.server_address[1]
    threads = [threading.Thread(target=srv.serve_forever, daemon=True) for srv in (site, internal)]
    for t in threads:
        t.start()
    yield s
    for srv in (site, internal):
        srv.shutdown()
        srv.server_close()


@pytest.fixture
async def session(servers: Servers, tmp_path: Path) -> AsyncIterator[BrowserSession]:
    names: dict[str, list[IPAddress]] = {
        "site.test": [LOOPBACK],
        "outside.test": [LOOPBACK],
        "internal.test": [LOOPBACK],
        "metadata.test": [ipaddress.ip_address("169.254.169.254")],
    }

    async def resolve(host: str, port: int) -> list[IPAddress]:
        if host in names:
            return names[host]
        raise OSError("NXDOMAIN")

    def block(ip: IPAddress, port: int) -> str | None:
        # Only the stand-in public site is reachable on loopback; the internal service (also on
        # loopback) is judged by the real policy and refused.
        if ip == LOOPBACK and port == servers.site_port:
            return None
        return default_block(ip, port)

    egress = EgressProxy(resolve=resolve, block=block)
    pool = BrowserPool(make_settings(tmp_path, block_resources=False), egress=egress)
    try:
        await pool.start()
    except Exception as err:
        pytest.skip(f"Chromium unavailable: {err}")
    servers.internal_hits.clear()
    s = await pool.open_session(f"r-{secrets.token_hex(3)}", allowed_domains=["site.test"])
    try:
        r = await s.execute(Navigate(url=servers.site("/home")))
        assert r.ok, r.message
        yield s
    finally:
        await pool.stop()


async def test_navigate_redirect_out_of_allowlist_is_undone(
    session: BrowserSession, servers: Servers
) -> None:
    r = await session.execute(Navigate(url=servers.bounce(servers.site("/evil", "outside.test"))))
    assert not r.ok
    assert "redirect to http://outside.test" in r.message and "blocked" in r.message
    obs = await session.observe()
    assert obs.url == servers.site("/home") and obs.title == "Home"
    assert "Evil button" not in [e.name for e in obs.elements]
    assert any("redirected outside the allowed domains" in b for b in obs.blocked)


async def test_click_redirect_out_of_allowlist_is_undone(
    session: BrowserSession, servers: Servers
) -> None:
    obs = await session.observe()
    r = await session.execute(Click(index=find(obs, "Go outside")))
    assert "redirect to http://outside.test" in r.message, r.message
    obs = await session.observe()
    assert obs.url == servers.site("/home")


async def test_redirect_to_internal_service_never_connects(
    session: BrowserSession, servers: Servers
) -> None:
    r = await session.execute(Navigate(url=servers.bounce(servers.internal("/secret"))))
    assert not r.ok and "blocked" in r.message
    obs = await session.observe()
    assert obs.url == servers.site("/home")
    r = await session.execute(Click(index=find(obs, "Go internal")))
    assert "blocked" in r.message
    assert servers.internal_hits == []  # the request was refused before any connection


async def test_redirect_to_metadata_address_never_connects(
    session: BrowserSession, servers: Servers
) -> None:
    r = await session.execute(Navigate(url=servers.bounce("http://metadata.test/latest/")))
    assert not r.ok and "blocked" in r.message
    assert (await session.current_page()).url == servers.site("/home")


async def test_subresources_to_internal_are_refused(
    session: BrowserSession, servers: Servers
) -> None:
    r = await session.execute(Navigate(url=servers.site("/subresources")))
    assert r.ok, r.message
    r = await session.execute(Extract(query="anything"))  # give the page time to fire
    assert servers.internal_hits == []
    obs = await session.observe()
    assert any("internal.test" in b and "private or reserved" in b for b in obs.blocked)


async def test_iframe_redirected_outside_is_not_observed(
    session: BrowserSession, servers: Servers
) -> None:
    r = await session.execute(Navigate(url=servers.site("/framed")))
    assert r.ok, r.message
    obs = await session.observe()
    names = [e.name for e in obs.elements]
    assert "Own button" in names
    assert "Framed evil" not in names


async def test_popup_redirected_outside_is_closed(
    session: BrowserSession, servers: Servers
) -> None:
    obs = await session.observe()
    r = await session.execute(Click(index=find(obs, "Popup outside")))
    assert "blocked" in r.message, r.message
    obs = await session.observe()
    assert obs.tabs == 1 and obs.url == servers.site("/home")


@pytest.mark.live
async def test_live_redirects_through_a_real_site(tmp_path: Path) -> None:
    """The original bypass: httpbin.org (allowed) bouncing elsewhere."""
    pool = BrowserPool(make_settings(tmp_path))
    await pool.start()
    try:
        s = await pool.open_session("live-redirect", allowed_domains=["httpbin.org"])
        await s.execute(Navigate(url="https://httpbin.org/html"))
        for target in ["https://example.com/", "http://169.254.169.254/latest/meta-data/"]:
            url = "https://httpbin.org/redirect-to?url=" + quote(target, safe="")
            r = await s.execute(Navigate(url=url))
            assert not r.ok and "blocked" in r.message, r.message
            assert (await s.current_page()).url == "https://httpbin.org/html"
        assert pool.egress is not None
        assert any("169.254.169.254" in target for _, target, _ in pool.egress.blocked)
    finally:
        await pool.stop()
