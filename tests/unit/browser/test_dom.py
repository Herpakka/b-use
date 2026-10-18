"""DOM extractor on the five fixture shapes: static, SPA, iframe, shadow DOM, infinite scroll."""

import pytest

from browser.actions import Click, Scroll
from browser.session import BrowserSession
from tests.browser_utils import find, goto

pytestmark = pytest.mark.browser


async def test_static_page(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    names = [e.name for e in obs.elements]

    assert obs.title == "Static fixture"
    assert "Home" in names and "Other site" in names
    assert "Close banner" in names  # aria-label wins over the "×" glyph
    assert "Fancy card" in names  # addEventListener click handler found via cursor:pointer
    q = obs.element(find(obs, "Search products"))
    assert q is not None and q.tag == "input" and q.placeholder == "Type a product"
    size = obs.element(find(obs, "Size"))
    assert size is not None and size.options == ["Small", "Medium", "Large"]
    assert size.value == "Small"
    agree = next(e for e in obs.elements if e.type == "checkbox")
    assert agree.checked is False
    # hidden, invisible, type=hidden and far-off-screen elements are excluded
    for absent in ("Hidden button", "Invisible button", "Far below"):
        assert absent not in names
    assert not any(e.type == "hidden" for e in obs.elements)
    # page text is interleaved, element text is not duplicated as page text
    rendered = obs.render()
    assert "Welcome to the fixture shop" in rendered
    assert "Free shipping over ฿1,000." in rendered
    assert obs.scroll.below > 3000
    # meaning carried only by attributes is surfaced as bracketed text
    assert "[image: Blue kettle photo]" in rendered
    assert "[star-rating Four]" in rendered
    assert "[4.5 out of 5 stars]" in rendered


async def test_password_values_are_masked(session: BrowserSession) -> None:
    obs = await goto(session, "static.html")
    page = await session.current_page()
    await page.fill("#pw", "hunter2")
    obs = await session.observe()
    pw = next(e for e in obs.elements if e.type == "password")
    assert pw.value == "••••"
    assert "hunter2" not in obs.render()


async def test_spa_waits_for_render_and_follows_push_state(session: BrowserSession) -> None:
    obs = await goto(session, "spa.html")
    # settle() waited past the 400 ms "Loading…" phase
    assert [e.name for e in obs.elements] == [
        "Open Alpha",
        "Open Beta",
        "Open Gamma",
        "Open Delta",
        "Open Epsilon",
    ]
    result = await session.execute(Click(index=find(obs, "Open Gamma")))
    assert result.ok, result.message
    obs = await session.observe()
    assert obs.url.endswith("/spa.html/gamma")
    assert "Gamma detail" in obs.render()
    assert "Back to list" in [e.name for e in obs.elements]  # second render wave included


async def test_iframes_cross_origin_visible_only(session: BrowserSession) -> None:
    obs = await goto(session, "iframe.html")
    child = [e for e in obs.elements if e.frame]
    assert [e.name for e in child] == ["Child button", "Child link"]
    assert all(e.frame and e.frame.startswith("https://other.test/") for e in child)
    assert "--- iframe https://other.test/iframe_child.html ---" in obs.render()
    # indices are unique and continue after the parent frame's elements
    indices = [e.index for e in obs.elements]
    assert indices == sorted(set(indices))

    result = await session.execute(Click(index=find(obs, "Child button")))
    assert result.ok, result.message
    obs = await session.observe()
    assert "Child clicked" in [e.name for e in obs.elements]


async def test_shadow_dom_and_slots(session: BrowserSession) -> None:
    obs = await goto(session, "shadow.html")
    names = [e.name for e in obs.elements]
    assert "Shadow button" in names
    assert "Slotted link" in names
    assert any(e.placeholder == "Shadow input" for e in obs.elements)
    assert "Card text inside shadow root" in obs.render()

    result = await session.execute(Click(index=find(obs, "Shadow button")))
    assert result.ok, result.message
    page = await session.current_page()
    assert await page.text_content("#status") == "shadow clicked"


async def test_infinite_scroll_loads_more(session: BrowserSession) -> None:
    obs = await goto(session, "infinite_scroll.html")
    first = {e.name for e in obs.elements}
    assert "Item 1" in first and "Item 40" not in first

    for _ in range(3):
        result = await session.execute(Scroll(direction="down", pages=2))
        assert result.ok, result.message
    obs = await session.observe()
    names = {e.name for e in obs.elements}
    assert "Item 1" not in names  # scrolled out of the observation window
    assert any(n.startswith("Item ") and int(n.split()[1]) > 40 for n in names)
    assert obs.scroll.above > 0


async def test_observation_tokens_rotate(session: BrowserSession) -> None:
    a = await goto(session, "static.html")
    b = await session.observe()
    assert a.dom_version != b.dom_version
    page = await session.current_page()
    count = await page.locator(f'[data-bu-idx^="{b.dom_version}-"]').count()
    assert count == len(b.elements)


async def test_field_and_form_kinds_for_the_approval_policy(session: BrowserSession) -> None:
    obs = await goto(session, "checkout.html")
    by_name = {e.name: e for e in obs.elements}
    assert (by_name["Card number"].field, by_name["Card number"].form) == ("payment", "payment")
    assert by_name["Name on card"].field is None and by_name["Name on card"].form == "payment"
    assert by_name["Continue"].form == "payment"
    assert by_name["Search"].form == "search" and by_name["Go"].form == "search"
    assert by_name["Password"].field == "password" and by_name["Sign in"].form == "login"
    assert by_name["Back to shop"].form is None
    assert "payment" not in obs.render()  # policy data only; the prompt is unchanged
