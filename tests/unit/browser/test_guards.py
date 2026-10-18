import ipaddress

import pytest

from browser.guards import (
    domain_allowed,
    host_reason,
    ip_reason,
    navigation_reason,
    normalize_domain,
    normalize_domains,
)

ALLOWED = normalize_domains(["Example.com", "*.shop.test", "https://books.toscrape.com/"])


def test_normalize_domains() -> None:
    assert ALLOWED == ("books.toscrape.com", "example.com", "shop.test")
    assert normalize_domain("ไทย.example") == "xn--o3cw4h.example"


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("example.com", True),
        ("www.example.com", True),
        ("a.b.shop.test", True),
        ("badexample.com", False),
        ("example.com.evil.net", False),
    ],
)
def test_domain_allowed_includes_subdomains_only(host: str, allowed: bool) -> None:
    assert domain_allowed(host, ALLOWED) is allowed


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/path",
        "http://www.example.com",
        "https://books.toscrape.com/catalogue/x.html",
        "about:blank",
    ],
)
def test_navigation_allowed(url: str) -> None:
    assert navigation_reason(url, ALLOWED) is None


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        ("https://evil.net/", "not in the allowed domains"),
        ("file:///etc/passwd", "scheme file"),
        ("javascript:alert(1)", "scheme javascript"),
        ("chrome://settings", "scheme chrome"),
        ("data:text/html,hi", "scheme data"),
        ("http://localhost:8000/", "internal host"),
        ("http://api.localhost/", "internal host"),
        ("http://127.0.0.1/", "private or reserved"),
        ("http://10.1.2.3/", "private or reserved"),
        ("http://192.168.1.1/", "private or reserved"),
        ("http://169.254.169.254/latest/meta-data", "private or reserved"),
        ("http://[::1]/", "private or reserved"),
        ("http://[::ffff:10.0.0.1]/", "private or reserved"),
        ("http://metadata.google.internal/", "internal host"),
        ("https:///nohost", "no host"),
    ],
)
def test_navigation_blocked(url: str, fragment: str) -> None:
    reason = navigation_reason(url, ALLOWED)
    assert reason is not None and fragment in reason


def test_host_reason_ignores_public_names_and_ips() -> None:
    assert host_reason("example.com") is None
    assert host_reason("93.184.216.34") is None


def test_ip_reason_unwraps_ipv4_mapped() -> None:
    assert ip_reason(ipaddress.ip_address("::ffff:127.0.0.1")) is not None
    assert ip_reason(ipaddress.ip_address("8.8.8.8")) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/out?url=https://evil.net/?d=secret",
        "https://example.com/out?url=https%3A%2F%2Fevil.net%2F",
        "https://example.com/out?u=https%253A%252F%252Fevil.net",  # encoded twice
        "https://example.com/out?next=//evil.net/x",
        "https://example.com/out?next=/%5Cevil.net",  # "/\evil.net" = "//evil.net"
        "https://example.com/out?next=///evil.net",
        "https://example.com/r#https://evil.net/",
        "https://example.com/out?to=http://169.254.169.254/latest/",
        "https://example.com/out?to=http://www.example.com@evil.net/",
    ],
)
def test_urls_embedding_outside_links_are_refused(url: str) -> None:
    reason = navigation_reason(url, ALLOWED)
    assert reason is not None and "embeds a link to" in reason


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/login?next=/account",
        "https://example.com/login?return=https://www.example.com/cart",
        "https://example.com/r?u=https://books.toscrape.com/",
        "https://example.com/app#/settings/profile",
        "https://example.com/search?q=what+is+https",
    ],
)
def test_urls_embedding_allowed_links_pass(url: str) -> None:
    assert navigation_reason(url, ALLOWED) is None
