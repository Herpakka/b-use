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
