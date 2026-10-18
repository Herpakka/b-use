"""Network policy enforced in the browser's request router — never in the prompt.

* Navigations (main frame and iframes) must be http(s) and on the run's allow-list.
* Every request — navigation or subresource — is refused if it targets localhost, a private /
  link-local / reserved IP, or a cloud metadata endpoint (SSRF).
* Images, fonts and media are optionally aborted to cut load time.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Request, Route, WebSocketRoute

from core.logging import get_logger

log = get_logger("browser.guards")

BLOCKABLE_RESOURCES = frozenset({"image", "font", "media"})
_METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata", "instance-data"})
_DNS_TTL_S = 60.0


def normalize_domain(domain: str) -> str:
    d = domain.strip().lower().rstrip(".")
    d = d.removeprefix("*.").removeprefix(".")
    if "://" in d:
        d = urlsplit(d).hostname or ""
    try:
        return d.encode("idna").decode("ascii")
    except UnicodeError:
        return d


def normalize_domains(domains: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({n for d in domains if (n := normalize_domain(d))}))


def _host(url: str) -> str | None:
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    try:
        return host.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError:
        return host.lower()


def ip_reason(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if not ip.is_global or ip.is_multicast:
        return f"private or reserved address {ip}"
    return None


def host_reason(host: str) -> str | None:
    """SSRF check on a hostname without DNS: localhost names, metadata names, IP literals."""
    if host == "localhost" or host.endswith(".localhost") or host in _METADATA_HOSTS:
        return f"internal host {host}"
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None
    return ip_reason(ip)


def domain_allowed(host: str, allowed: Iterable[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in allowed)


def navigation_reason(url: str, allowed: Iterable[str]) -> str | None:
    """Why a navigation to ``url`` must be refused, or None. Pure: no DNS."""
    if url == "about:blank":
        return None
    scheme = urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        return f"scheme {scheme or '(none)'} is not allowed"
    host = _host(url)
    if host is None:
        return "URL has no host"
    if reason := host_reason(host):
        return reason
    if not domain_allowed(host, allowed):
        return f"{host} is not in the allowed domains"
    return None


@dataclass
class NetworkPolicy:
    allowed_domains: tuple[str, ...]
    block_resources: bool = True
    resolve_dns: bool = True
    on_blocked: Callable[[str, str], None] | None = None
    _dns_cache: dict[str, tuple[float, str | None]] = field(default_factory=dict, repr=False)

    async def dns_reason(self, host: str) -> str | None:
        """Resolve ``host`` and refuse it if any address is internal (cached briefly)."""
        now = time.monotonic()
        cached = self._dns_cache.get(host)
        if cached and now - cached[0] < _DNS_TTL_S:
            return cached[1]
        reason: str | None = None
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, None, type=socket.SOCK_STREAM
            )
        except OSError:
            infos = []  # let Chromium report the DNS failure itself
        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(str(addr).split("%")[0])
            except ValueError:
                continue
            if r := ip_reason(ip):
                reason = f"{host} resolves to {r}"
                break
        self._dns_cache[host] = (now, reason)
        return reason

    def _blocked(self, url: str, reason: str) -> None:
        log.info("request_blocked", url=url[:200], reason=reason)
        if self.on_blocked is not None:
            self.on_blocked(url, reason)


async def install_routes(context: BrowserContext, policy: NetworkPolicy) -> None:
    """One handler for every request in the context: SSRF block, allow-list, resource blocking.

    Must run before any page is created. Service workers must be blocked on the context
    (``service_workers="block"``) or their requests bypass this router.
    """

    async def handle(route: Route, request: Request) -> None:
        url = request.url
        if url.startswith(("data:", "blob:")):
            await route.continue_()
            return
        host = _host(url)
        if request.is_navigation_request():
            reason = navigation_reason(url, policy.allowed_domains)
            if reason is None and policy.resolve_dns and host:
                reason = await policy.dns_reason(host)
            if reason is not None:
                policy._blocked(url, reason)
                # A 204 makes the browser cancel the navigation and keep the current page;
                # aborting would leave the tab on Chrome's error page.
                await route.fulfill(status=204, body="")
                return
        elif host and (reason := host_reason(host)) is not None:
            await route.abort("blockedbyclient")
            return
        if policy.block_resources and request.resource_type in BLOCKABLE_RESOURCES:
            await route.abort("blockedbyclient")
            return
        await route.continue_()

    async def handle_ws(ws: WebSocketRoute) -> None:
        host = _host(ws.url)
        reason = host_reason(host) if host else "no host"
        if reason is None and policy.resolve_dns and host:
            reason = await policy.dns_reason(host)
        if reason is not None:
            log.info("websocket_blocked", url=ws.url[:200], reason=reason)
            await ws.close(code=1008, reason="blocked")
            return
        ws.connect_to_server()

    await context.route(re.compile(r".*"), handle)
    await context.route_web_socket(re.compile(r".*"), handle_ws)
