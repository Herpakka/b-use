"""The egress proxy on its own: plain HTTP and CONNECT, auth, and what it refuses."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from browser.egress import EgressProxy, IPAddress, default_block, split_host_port

LOOPBACK = ipaddress.ip_address("127.0.0.1")


@dataclass
class Origin:
    """A tiny HTTP server standing in for a public site; records what it received."""

    port: int = 0
    heads: list[str] = field(default_factory=list)
    server: asyncio.Server | None = None

    async def start(self) -> None:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
            self.heads.append(head)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello")
            await writer.drain()
            writer.close()

        self.server = await asyncio.start_server(handle, "127.0.0.1", 0)
        self.port = int(self.server.sockets[0].getsockname()[1])


@pytest.fixture
async def setup() -> AsyncIterator[tuple[EgressProxy, Origin]]:
    origin = Origin()
    await origin.start()
    names: dict[str, list[IPAddress]] = {
        "site.test": [LOOPBACK],
        "internal.test": [ipaddress.ip_address("10.1.2.3")],
        "mixed.test": [ipaddress.ip_address("93.184.216.34"), ipaddress.ip_address("10.0.0.9")],
    }

    async def resolve(host: str, port: int) -> list[IPAddress]:
        if host not in names:
            raise OSError("NXDOMAIN")
        return names[host]

    def block(ip: IPAddress, port: int) -> str | None:
        # The origin on loopback plays the public site; every other address is judged as real.
        return None if ip == LOOPBACK and port == origin.port else default_block(ip, port)

    proxy = EgressProxy(resolve=resolve, block=block)
    await proxy.start()
    yield proxy, origin
    await proxy.stop()
    assert origin.server is not None
    origin.server.close()


def auth(proxy: EgressProxy, user: str = "acme.run-1") -> str:
    token = base64.b64encode(f"{user}:{proxy.password}".encode()).decode()
    return f"Proxy-Authorization: Basic {token}\r\n"


async def exchange(proxy: EgressProxy, request: str) -> str:
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
    writer.write(request.encode("latin-1"))
    await writer.drain()
    data = await asyncio.wait_for(reader.read(), timeout=5)
    writer.close()
    return data.decode("latin-1")


async def test_plain_http_is_forwarded_without_proxy_headers(setup) -> None:
    proxy, origin = setup
    url = f"http://site.test:{origin.port}/hello?x=1"
    res = await exchange(
        proxy,
        f"GET {url} HTTP/1.1\r\nHost: site.test:{origin.port}\r\n{auth(proxy)}"
        "Proxy-Connection: keep-alive\r\n\r\n",
    )
    assert res.startswith("HTTP/1.1 200 OK") and res.endswith("hello")
    head = origin.heads[0]
    assert head.startswith("GET /hello?x=1 HTTP/1.1\r\n")  # origin-form
    assert "Proxy-Authorization" not in head and "Proxy-Connection" not in head
    assert "Connection: close" in head and f"Host: site.test:{origin.port}" in head


async def test_connect_tunnel_to_allowed_target(setup) -> None:
    proxy, origin = setup
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
    writer.write(f"CONNECT site.test:{origin.port} HTTP/1.1\r\n{auth(proxy)}\r\n".encode())
    await writer.drain()
    status = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
    assert status.startswith(b"HTTP/1.1 200")
    writer.write(b"GET /in-tunnel HTTP/1.1\r\nHost: site.test\r\n\r\n")
    await writer.drain()
    body = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    assert body.endswith(b"hello")
    assert origin.heads[0].startswith("GET /in-tunnel ")


async def test_requires_the_context_credentials(setup) -> None:
    proxy, origin = setup
    res = await exchange(proxy, f"GET http://site.test:{origin.port}/ HTTP/1.1\r\n\r\n")
    assert res.startswith("HTTP/1.1 407") and "Proxy-Authenticate: Basic" in res
    bad = base64.b64encode(b"acme.run-1:wrong").decode()
    res = await exchange(
        proxy,
        f"GET http://site.test:{origin.port}/ HTTP/1.1\r\nProxy-Authorization: Basic {bad}\r\n\r\n",
    )
    assert res.startswith("HTTP/1.1 407")
    assert origin.heads == []


@pytest.mark.parametrize(
    ("target", "fragment"),
    [
        ("internal.test:443", "internal.test resolves to private or reserved address 10.1.2.3"),
        ("mixed.test:443", "mixed.test resolves to private"),  # any internal answer refuses
        ("10.0.0.1:443", "private or reserved address 10.0.0.1"),
        ("169.254.169.254:80", "private or reserved address 169.254.169.254"),
        ("[::1]:443", "private or reserved address ::1"),
        ("localhost:443", "internal host localhost"),
        ("api.localhost:443", "internal host api.localhost"),
        ("metadata.google.internal:80", "internal host metadata.google.internal"),
    ],
)
async def test_connect_refuses_internal_targets(setup, target: str, fragment: str) -> None:
    proxy, _ = setup
    res = await exchange(proxy, f"CONNECT {target} HTTP/1.1\r\n{auth(proxy)}\r\n")
    assert res.startswith("HTTP/1.1 403"), res
    assert fragment in res
    user, _target, reason = proxy.blocked[-1]
    assert user == "acme.run-1" and fragment in reason


async def test_plain_http_to_internal_is_refused(setup) -> None:
    proxy, _ = setup
    res = await exchange(proxy, f"GET http://internal.test/secret HTTP/1.1\r\n{auth(proxy)}\r\n")
    assert res.startswith("HTTP/1.1 403")


async def test_unresolvable_and_malformed(setup) -> None:
    proxy, _ = setup
    res = await exchange(proxy, f"CONNECT nowhere.test:443 HTTP/1.1\r\n{auth(proxy)}\r\n")
    assert res.startswith("HTTP/1.1 502") and "cannot resolve nowhere.test" in res
    res = await exchange(proxy, f"GET /relative HTTP/1.1\r\n{auth(proxy)}\r\n")
    assert res.startswith("HTTP/1.1 400")
    res = await exchange(proxy, f"GET ftp://site.test/ HTTP/1.1\r\n{auth(proxy)}\r\n")
    assert res.startswith("HTTP/1.1 400")


def test_split_host_port() -> None:
    assert split_host_port("example.com:8443", 443) == ("example.com", 8443)
    assert split_host_port("Example.COM.", 443) == ("example.com", 443)
    assert split_host_port("[::1]:443", 80) == ("::1", 443)
    assert split_host_port("::1", 443) == ("::1", 443)


def test_context_proxy_settings() -> None:
    proxy = EgressProxy()
    settings = proxy.context_proxy("acme.run-1", "http://127.0.0.1:3128")
    assert settings["bypass"] == "<-loopback>"  # loopback must go through the proxy too
    assert settings["username"] == "acme.run-1" and settings["password"] == proxy.password
    assert len(proxy.password) > 20
