"""{{SECRET:name}} placeholders through real Chromium: the value goes into the field and never
comes back out — not in results, observations, URLs or Extract text."""

import pytest

from browser.actions import Extract, Navigate, TypeText
from browser.session import BrowserSession
from core.secrets import Secret
from tests.browser_utils import find, goto

pytestmark = pytest.mark.browser

USER = Secret("user", "alice@example.com", ("fixtures.test",))
PW = Secret("pw", "correct-horse-battery", ("fixtures.test",))
SECRETS = {"user": USER, "pw": PW}


async def test_secret_is_typed_but_never_read_back(session: BrowserSession) -> None:
    obs = await goto(session, "login.html")
    pw = next(e.index for e in obs.elements if e.type == "password")
    r = await session.execute(
        TypeText(index=find(obs, "User"), text="{{SECRET:user}}"), secrets=SECRETS
    )
    assert r.ok, r.message
    assert r.message == f'type "{{{{SECRET:user}}}}" into [{find(obs, "User")}] "User" → ok'
    page = await session.current_page()
    assert await page.input_value("#user") == "alice@example.com"  # the real value went in

    rendered = (await session.observe()).render()
    assert "alice@example.com" not in rendered
    assert 'value="{{SECRET:user}}"' in rendered

    # A GET form: both values end up in the URL, and the user name on the page.
    r = await session.execute(
        TypeText(index=pw, text="{{SECRET:pw}}", submit=True), secrets=SECRETS
    )
    assert r.ok, r.message
    obs = await session.observe()
    assert "correct-horse" not in obs.url and "alice" not in obs.url
    assert "pw={{SECRET:pw}}" in obs.url
    assert "Welcome back, {{SECRET:user}}!" in obs.render()
    r = await session.execute(Extract(query="who is logged in"))
    assert r.extracted_text is not None and "{{SECRET:user}}" in r.extracted_text
    assert "alice@example.com" not in r.extracted_text and "alice" not in r.url
    assert "alice" not in session.info().url


async def test_secret_refused_outside_its_domains(session: BrowserSession) -> None:
    obs = await goto(session, "login.html")
    verify = find(obs, "Verify your account")  # an iframe from other.test (an allowed domain)
    r = await session.execute(TypeText(index=verify, text="{{SECRET:pw}}"), secrets=SECRETS)
    assert not r.ok
    assert "may only be typed on fixtures.test, not on other.test" in r.message
    frame = next(f for f in (await session.current_page()).frames if "other.test" in f.url)
    assert await frame.input_value("#verify") == ""


async def test_unusable_placeholders_are_refused(session: BrowserSession) -> None:
    obs = await goto(session, "login.html")
    r = await session.execute(
        TypeText(index=find(obs, "User"), text="{{SECRET:nope}}"), secrets=SECRETS
    )
    assert not r.ok and "unknown secret 'nope'" in r.message
    r = await session.execute(TypeText(index=find(obs, "User"), text="{{SECRET:user}}"))
    assert not r.ok and "unknown secret 'user'" in r.message  # no values handed over
    r = await session.execute(
        TypeText(index=find(obs, "Plan"), text="{{SECRET:user}}"), secrets=SECRETS
    )
    assert not r.ok and "text fields only" in r.message
    r = await session.execute(
        Navigate(url="https://fixtures.test/login.html?user={{SECRET:user}}"), secrets=SECRETS
    )
    assert not r.ok and "secrets can only be typed into fields" in r.message
    page = await session.current_page()
    assert await page.input_value("#user") == ""
