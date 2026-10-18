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
from urllib.parse import parse_qsl, unquote, urlsplit

from playwright.async_api import BrowserContext, Request, Route, WebSocketRoute

from core.domains import domain_allowed, normalize_domain, normalize_domains
from core.domains import url_host as _host
from core.logging import get_logger

__all__ = [
    "NetworkPolicy",
    "domain_allowed",
    "embedded_url_reason",
    "host_reason",
    "install_routes",
    "ip_reason",
    "navigation_reason",
    "normalize_domain",
    "normalize_domains",
]

log = get_logger("browser.guards")

BLOCKABLE_RESOURCES = frozenset({"image", "font", "media"})
_METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata", "instance-data"})
_DNS_TTL_S = 60.0


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
    return embedded_url_reason(url, allowed)


# Browsers read a backslash as "/" and ignore extra slashes in http(s) URLs ("/\evil", "///evil").
_EMBEDDED = re.compile(r"^(?:https?:)?/{2,}([^/?#\s]+)", re.IGNORECASE)


def embedded_url_reason(url: str, allowed: Iterable[str]) -> str | None:
    """Refuse URLs that carry a link to a non-allowed host in their query or fragment.

    That is the shape of an open redirect (``allowed.site/out?url=https://elsewhere/?data=…``):
    the router never sees redirect hops, so without this check the request to the other host
    would leave the browser — together with whatever the URL carries — before the allow-list
    check after the redirect could undo the navigation.
    """
    parts = urlsplit(url)
    allowed = tuple(allowed)
    for raw in (parts.query, parts.fragment):
        if not raw:
            continue
        # The whole string too: "#https://…" or "?https://…" carry the link without a key.
        values = [raw, *(v for _, v in parse_qsl(raw, keep_blank_values=True))]
        for value in values:
            for _ in range(3):  # links are often encoded more than once
                link = value.strip().replace("\\", "/")
                if (m := _EMBEDDED.match(link)) and (host := _host("http://" + m[1])):
                    if host_reason(host) or not domain_allowed(host, allowed):
                        return f"URL embeds a link to {host}, outside the allowed domains"
                    break
                decoded = unquote(value)
                if decoded == value:
                    break
                value = decoded
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
