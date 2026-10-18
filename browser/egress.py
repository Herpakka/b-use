"""Egress proxy: the network-level SSRF guard that every browser request passes through.

The request router (guards.py) cannot see redirect hops — Playwright does not route requests
that follow a redirect — and it does not resolve the hostnames of subresource requests. This
proxy covers all traffic: it resolves each target itself, refuses loopback / private /
link-local / reserved / metadata addresses, and then connects to exactly the address it
checked, so a DNS answer cannot change between the check and the connection (rebinding).

It speaks just enough of the HTTP proxy protocol: CONNECT tunnels (HTTPS, WSS) and
absolute-form plain HTTP. Each browser context authenticates with its session id as the user
name, which attributes every block to a session and keeps other processes from using it.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import ipaddress
import secrets
import socket
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TypedDict
from urllib.parse import urlsplit

from browser.guards import ip_reason
from core.logging import get_logger

log = get_logger("browser.egress")

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], Awaitable[list[IPAddress]]]
BlockCheck = Callable[[IPAddress, int], str | None]
BlockSink = Callable[[str | None, str, str], None]  # (session id, target, reason)

_INTERNAL_NAMES = frozenset({"localhost", "metadata.google.internal", "metadata", "instance-data"})
_HOP_HEADERS = frozenset({"proxy-authorization", "proxy-connection", "connection", "keep-alive"})
_MAX_HEAD = 64 * 1024


class ProxySettings(TypedDict):
    server: str
    bypass: str
    username: str
    password: str


def context_proxy(session_id: str, server_url: str, password: str) -> ProxySettings:
    """Playwright ``proxy=`` settings for one browser context."""
    # "<-loopback>" removes Chromium's implicit bypass for localhost, so even loopback
    # requests pass through the proxy (and are refused there).
    return ProxySettings(
        server=server_url, bypass="<-loopback>", username=session_id, password=password
    )


class _RefusedError(Exception):
    """The target is not allowed (403)."""


class _UnreachableError(Exception):
    """The target could not be resolved or connected to (502)."""


async def system_resolve(host: str, port: int) -> list[IPAddress]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    out: list[IPAddress] = []
    for info in infos:
        with contextlib.suppress(ValueError):
            out.append(ipaddress.ip_address(str(info[4][0]).split("%")[0]))
    return out


def default_block(ip: IPAddress, port: int) -> str | None:
    return ip_reason(ip)


def split_host_port(authority: str, default_port: int) -> tuple[str, int]:
    if authority.startswith("["):  # [v6]:port
        host, _, rest = authority[1:].partition("]")
        port = rest.removeprefix(":")
    elif authority.count(":") == 1:
        host, _, port = authority.partition(":")
    else:  # bare host, or an unbracketed IPv6 literal
        host, port = authority, ""
    return host.lower().rstrip("."), int(port) if port.isdigit() else default_port


def _response(status: str, reason: str, extra: str = "") -> bytes:
    body = reason.encode()
    return (
        f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
        f"X-BU-Blocked: {reason[:200]}\r\n{extra}Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin-1", "replace") + body


async def _copy(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    try:
        while data := await src.read(65536):
            dst.write(data)
            await dst.drain()
        if dst.can_write_eof():
            dst.write_eof()
    except (ConnectionError, OSError):
        pass


class EgressProxy:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        resolve: Resolver = system_resolve,
        block: BlockCheck = default_block,
        on_blocked: BlockSink | None = None,
        connect_timeout_s: float = 10.0,
        password: str | None = None,
        require_auth: bool = True,
    ) -> None:
        self.host = host
        self._port = port
        self._resolve = resolve
        self._block = block
        self.on_blocked = on_blocked
        self.connect_timeout_s = connect_timeout_s
        self.password = (password or secrets.token_urlsafe(24)) if require_auth else ""
        self.blocked: deque[tuple[str | None, str, str]] = deque(maxlen=200)
        self._server: asyncio.Server | None = None

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, self.host, self._port, limit=_MAX_HEAD
        )
        self._port = int(self._server.sockets[0].getsockname()[1])
        log.info("egress_proxy_listening", host=self.host, port=self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None

    def context_proxy(self, session_id: str, server_url: str) -> ProxySettings:
        return context_proxy(session_id, server_url, self.password)

    # ------------------------------------------------------------------ connection handling
    def _user(self, headers: list[tuple[str, str]]) -> tuple[bool, str | None]:
        if not self.password:
            return True, None
        for name, value in headers:
            if name.lower() == "proxy-authorization" and value.lower().startswith("basic "):
                try:
                    user, _, password = base64.b64decode(value[6:].strip()).decode().partition(":")
                except (ValueError, UnicodeDecodeError):
                    return False, None
                if hmac.compare_digest(password.encode(), self.password.encode()):
                    return True, user
        return False, None

    def _record(self, user: str | None, target: str, reason: str) -> None:
        log.info("egress_blocked", session_id=user, target=target[:200], reason=reason)
        self.blocked.append((user, target, reason))
        if self.on_blocked is not None:
            self.on_blocked(user, target, reason)

    async def _open(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            literal: IPAddress | None = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            literal = None
        if literal is None and (host in _INTERNAL_NAMES or host.endswith(".localhost")):
            raise _RefusedError(f"internal host {host}")
        if literal is not None:
            addresses = [literal]
        else:
            try:
                addresses = await self._resolve(host, port)
            except OSError as err:
                raise _UnreachableError(f"cannot resolve {host}") from err
        if not addresses:
            raise _UnreachableError(f"cannot resolve {host}")
        for ip in addresses:  # refuse if *any* answer is internal: no round-robin roulette
            if reason := self._block(ip, port):
                raise _RefusedError(
                    reason if literal is not None else f"{host} resolves to {reason}"
                )
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(str(addresses[0]), port), self.connect_timeout_s
            )
        except (OSError, TimeoutError) as err:
            raise _UnreachableError(f"cannot connect to {host}:{port}") from err

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream: asyncio.StreamWriter | None = None
        try:
            try:
                head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
                return
            lines = head.decode("latin-1").split("\r\n")
            parts = lines[0].split(" ")
            if len(parts) != 3:
                writer.write(_response("400 Bad Request", "malformed request line"))
                return
            method, target, version = parts
            headers = [
                (name.strip(), value.strip())
                for name, _, value in (line.partition(":") for line in lines[1:] if line)
            ]
            authorized, user = self._user(headers)
            if not authorized:
                writer.write(
                    _response(
                        "407 Proxy Authentication Required",
                        "proxy authentication required",
                        'Proxy-Authenticate: Basic realm="bu-egress"\r\n',
                    )
                )
                return

            connect = method.upper() == "CONNECT"
            if connect:
                host, port = split_host_port(target, 443)
                path = ""
            else:
                url = urlsplit(target)
                if url.scheme != "http" or not url.hostname:
                    writer.write(_response("400 Bad Request", "absolute http:// URL required"))
                    return
                host, port = url.hostname.lower().rstrip("."), url.port or 80
                path = (url.path or "/") + (f"?{url.query}" if url.query else "")

            try:
                upstream_reader, upstream = await self._open(host, port)
            except _RefusedError as err:
                self._record(user, f"{host}:{port}", str(err))
                writer.write(_response("403 Forbidden", str(err)))
                return
            except _UnreachableError as err:
                writer.write(_response("502 Bad Gateway", str(err)))
                return

            if connect:
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            else:
                kept = [(n, v) for n, v in headers if n.lower() not in _HOP_HEADERS]
                forwarded = "".join(f"{n}: {v}\r\n" for n, v in kept)
                # One request per upstream connection: Chromium may reuse its connection to the
                # proxy for another host, so the exchange must end with the response.
                upstream.write(
                    f"{method} {path} {version}\r\n{forwarded}Connection: close\r\n\r\n".encode(
                        "latin-1"
                    )
                )
            await writer.drain()
            await asyncio.gather(_copy(reader, upstream), _copy(upstream_reader, writer))
        except (ConnectionError, OSError):
            pass
        finally:
            for w in (upstream, writer):
                if w is not None:
                    w.close()
                    with contextlib.suppress(Exception):
                        await w.wait_closed()
