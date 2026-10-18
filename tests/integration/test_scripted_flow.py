"""M1: drive the browser through a whole search flow with a script — no LLM involved."""

import pytest

from browser.actions import Click, Extract, Navigate, TypeText
from browser.pool import BrowserPool
from browser.session import BrowserSession
from tests.browser_utils import find, goto

pytestmark = pytest.mark.browser


async def test_search_click_extract(session: BrowserSession) -> None:
    obs = await goto(session, "search/index.html")
    r = await session.execute(TypeText(index=find(obs, "Search"), text="kettle", submit=True))
    assert r.ok, r.message
    assert "/search/results.html?q=kettle" in r.url

    obs = await session.observe()
    assert [e.name for e in obs.elements] == ["Blue Kettle", "Red Kettle"]
    r = await session.execute(Click(index=find(obs, "Blue Kettle")))
    assert r.ok, r.message

    r = await session.execute(Extract(query="price"))
    assert r.ok and r.extracted_text is not None
    assert "Price: ฿1,290" in r.extracted_text
    assert "In stock (12)" in r.extracted_text


@pytest.mark.live
async def test_live_wikipedia_flow(pool: BrowserPool) -> None:
    s = await pool.open_session("live-wiki", allowed_domains=["wikipedia.org"])
    try:
        r = await s.execute(Navigate(url="https://en.wikipedia.org/wiki/Main_Page"))
        assert r.ok, r.message
        obs = await s.observe()
        search = next(e for e in obs.elements if e.tag == "input" and e.type in ("search", "text"))
        r = await s.execute(
            TypeText(index=search.index, text="Python (programming language)", submit=True)
        )
        assert r.ok, r.message
        obs = await s.observe()
        assert "Python" in obs.title
        r = await s.execute(Extract(query="first appeared"))
        assert r.extracted_text and "Guido van Rossum" in r.extracted_text
    finally:
        await pool.close_session("live-wiki", persist=False)
