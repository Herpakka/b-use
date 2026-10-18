import pytest

from browser.actions import Click, Done, Extract, Navigate, Scroll, TypeText
from browser.session import BrowserSession
from tests.browser_utils import find, goto

pytestmark = pytest.mark.browser


async def test_type_select_check_and_submit(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    q = find(obs, "Search products")
    r = await session.execute(TypeText(index=q, text="kettle"))
    assert r.ok and r.message == f'type "kettle" into [{q}] "Search products" → ok'

    r = await session.execute(TypeText(index=find(obs, "Size"), text="Large"))
    assert r.ok and r.message.endswith('→ ok (selected "Large")'), r.message
    r = await session.execute(Click(index=find(obs, "I agree")))
    assert r.ok, r.message

    obs = await session.observe()
    assert obs.element(find(obs, "Size")).value == "Large"  # type: ignore[union-attr]
    assert next(e for e in obs.elements if e.type == "checkbox").checked is True

    r = await session.execute(TypeText(index=find(obs, "Search products"), text="tea", submit=True))
    assert r.ok and r.message.endswith("+ Enter → ok")
    page = await session.current_page()
    assert await page.text_content("#status") == "submitted tea"


async def test_password_is_never_echoed(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    pw = next(e for e in obs.elements if e.type == "password")
    r = await session.execute(TypeText(index=pw.index, text="hunter2"))
    assert r.ok
    assert "hunter2" not in r.message and "••••" in r.message


async def test_click_select_is_redirected_to_type_text(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    r = await session.execute(Click(index=find(obs, "Size")))
    assert not r.ok and "type_text" in r.message


async def test_unknown_index(session: BrowserSession) -> None:
    await goto(session, "static.html")
    r = await session.execute(Click(index=999))
    assert not r.ok and not r.stale
    assert "no element [999]" in r.message


async def test_stale_when_target_changed(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    idx = find(obs, "Show alert")
    page = await session.current_page()
    await page.evaluate("document.getElementById('alert').textContent = 'Something else'")
    r = await session.execute(Click(index=idx))
    assert r.stale and not r.ok
    assert await page.text_content("#status") == "idle"  # nothing was clicked


async def test_stale_when_target_removed(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    idx = find(obs, "Show alert")
    page = await session.current_page()
    await page.evaluate("document.getElementById('alert').remove()")
    assert (await session.execute(Click(index=idx))).stale


async def test_stale_when_expected_version_differs(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    await session.observe()  # someone re-observed: indices from `obs` are no longer valid
    r = await session.execute(Click(index=find(obs, "Home")), expected_version=obs.dom_version)
    assert r.stale


async def test_unrelated_dom_churn_is_not_stale(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    page = await session.current_page()
    await page.evaluate("document.body.appendChild(document.createElement('div'))")
    r = await session.execute(Click(index=find(obs, "Fancy card")))
    assert r.ok, r.message
    assert await page.text_content("#status") == "card clicked"


async def test_dialog_is_dismissed_and_reported(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    r = await session.execute(Click(index=find(obs, "Show alert")))
    assert r.ok, r.message
    obs = await session.observe()
    assert obs.dialogs == ["alert: hello from fixture"]
    assert (await session.observe()).dialogs == []  # reported once


async def test_popup_becomes_active_tab(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    r = await session.execute(Click(index=find(obs, "Open in new tab")))
    assert r.ok and "(opened a new tab)" in r.message
    obs = await session.observe()
    assert obs.title == "Popup target"
    assert obs.tabs == 2


async def test_navigate_outside_allow_list_is_refused(session: BrowserSession) -> None:
    await goto(session, "static.html")
    for url, reason in [
        ("https://blocked.test/", "not in the allowed domains"),
        ("http://127.0.0.1:9/", "private or reserved"),
        ("http://169.254.169.254/latest/meta-data/", "private or reserved"),
        ("file:///etc/passwd", "scheme file"),
    ]:
        r = await session.execute(Navigate(url=url))
        assert not r.ok and reason in r.message, r.message
    assert (await session.current_page()).url == "https://fixtures.test/static.html"


async def test_link_to_disallowed_domain_is_blocked_by_router(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    r = await session.execute(Click(index=find(obs, "Blocked site")))
    assert "navigation blocked" in r.message
    obs = await session.observe()
    assert obs.url == "https://fixtures.test/static.html"
    assert len(obs.blocked) == 1 and obs.blocked[0].startswith("https://blocked.test/page")


async def test_allowed_cross_domain_link_works(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    r = await session.execute(Click(index=find(obs, "Other site")))
    assert r.ok and r.url == "https://other.test/iframe_child.html"


async def test_subresource_to_private_ip_is_aborted(session: BrowserSession) -> None:
    await goto(session, "static.html")
    page = await session.current_page()
    outcome = await page.evaluate(
        """() => fetch("http://169.254.169.254/latest/meta-data/")
            .then(() => "loaded").catch(() => "failed")"""
    )
    assert outcome == "failed"


async def test_scroll_extract_done(session: BrowserSession) -> None:
    await goto(session, "static.html")
    r = await session.execute(Scroll(direction="down", pages=1))
    assert r.ok and r.message.startswith("scroll down 1 page(s) → 800px above")
    r = await session.execute(Extract(query="shipping policy"))
    assert r.ok and r.extracted_text and "Free shipping over ฿1,000." in r.extracted_text
    assert (await session.execute(Done(result="x"))).ok


async def test_extract_text_includes_form_state_and_annotations(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    await session.execute(TypeText(index=find(obs, "Size"), text="Medium"))
    await session.execute(TypeText(index=find(obs, "Search products"), text="kettle"))
    await session.execute(Click(index=find(obs, "I agree")))
    r = await session.execute(Extract(query="form state"))
    text = r.extracted_text or ""
    assert '[select "Size": selected "Medium"]' in text
    assert '[field "Search products": "kettle"]' in text
    assert '[checkbox "I agree": checked]' in text
    assert "[star-rating Four]" in text and "[image: Blue kettle photo]" in text
    assert "Far below" in text  # the whole page, not just the viewport
    assert "Hidden button" not in text
    # read-only: the observation's element tags are untouched, so its indices still work
    r = await session.execute(
        Click(index=find(obs, "Fancy card")), expected_version=obs.dom_version
    )
    assert r.ok, r.message
