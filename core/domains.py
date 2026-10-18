"""Domain names as allow-lists use them: normalised to lowercase IDNA, subdomains included."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit


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


def url_host(url: str) -> str | None:
    """The URL's hostname, lowercase IDNA, or None."""
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


def domain_allowed(host: str, allowed: Iterable[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in allowed)
