"""The full graph on real Chromium with a scripted LLM: every layer except OpenAI is real."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.context import AgentContext
from agent.graph import compile_graph
from agent.runner import drive, initial_state, limits_from
from browser.local import LocalBrowserService
from browser.pool import BrowserPool
from core.costs import PriceTable
from tests.agent_fakes import PLAN, ScriptedChatModel, text, tool
from tests.browser_utils import serve_fixtures
from tests.conftest import make_settings

pytestmark = pytest.mark.browser


class FixtureBrowserService(LocalBrowserService):
    async def open(
        self, session_id: str, *, allowed_domains: Sequence[str], vision: bool = False
    ) -> None:
        await super().open(session_id, allowed_domains=allowed_domains, vision=vision)
        session = self.pool.get(session_id)
        assert session is not None
        await serve_fixtures(session.context)


async def test_search_extract_and_denied_order(pool: BrowserPool) -> None:
    approvals: list[dict[str, Any]] = []

    async def deny(payload: dict[str, Any]) -> dict[str, Any]:
        approvals.append(payload)
        return {"approved": False, "answer": "do not buy"}

    settings = make_settings(Path("unused"))
    decide = ScriptedChatModel(
        script=[
            PLAN,
            tool("TypeText", index=1, text="kettle", submit=True),  # search box on index.html
            tool("Click", index=1),  # "Blue Kettle" on results.html
            tool("Extract", query="price"),
            tool("Click", index=2),  # "Place order" → human gate
            tool("Done", result="Blue Kettle costs ฿1,290"),
        ]
    )
    ctx = AgentContext(
        browser=FixtureBrowserService(pool),
        llm_decide=decide,
        llm_small=ScriptedChatModel(script=[text("฿1,290")]),
        settings=settings,
        prices=PriceTable.from_settings(settings),
    )
    state = initial_state(
        ctx,
        task="Find the price of the blue kettle",
        allowed_domains=["fixtures.test"],
        start_url="https://fixtures.test/search/index.html",
        limits=limits_from(settings),
    )
    final, _ = await drive(
        compile_graph(InMemorySaver()), ctx, state["run_id"], state, max_steps=25, on_human=deny
    )

    assert final["status"] == "done", final["failure_reason"]
    assert final["history"][:3] == [
        '1. type "kettle" into [1] "Search" + Enter → ok',
        '2. click [1] "Blue Kettle" → ok',
        '3. extract "price" → ฿1,290',
    ]
    assert approvals and approvals[0]["action"] == 'Click [2] "Place order"'
    assert any(h.startswith("human denied") for h in final["history"])
    # The extract prompt carried the real page text, fenced.
    small = ctx.llm_small
    assert isinstance(small, ScriptedChatModel)
    assert "Price: ฿1,290" in str(small.calls[0][-1].content)
    # The decide prompt showed the real rendered observation of the detail page.
    detail_prompt = str(decide.calls[4][-1].content)
    assert "[2]<button>Place order</button>" in detail_prompt
    assert pool.get(final["session_id"]) is None  # finalize closed the session
