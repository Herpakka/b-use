"""A run paused for a human survives a full restart when checkpoints live in Postgres."""

from __future__ import annotations

import os
import secrets
import sys
from typing import Any

import pytest

from agent.graph import compile_graph
from agent.runner import drive, open_checkpointer
from tests.agent_fakes import PLAN, FakeBrowser, make_ctx, page, run_task, tool

pytestmark = pytest.mark.pg

HOME = "https://shop.test/"
SHOP = {HOME: page(HOME, ("button", "Place order"))}


@pytest.fixture
async def pg_url() -> str:
    if sys.platform == "win32":
        pytest.skip("async psycopg cannot use the Windows Proactor loop; run in the container")
    url = os.environ.get("BU_DATABASE_URL")
    if not url:
        pytest.skip("BU_DATABASE_URL not set")
    import psycopg

    try:
        conn = await psycopg.AsyncConnection.connect(url, connect_timeout=3)
        await conn.close()
    except Exception as err:
        pytest.skip(f"Postgres unreachable: {err}")
    return url


async def approve(payload: dict[str, Any]) -> dict[str, Any]:
    return {"approved": True}


async def test_resume_from_postgres_after_restart(pg_url: str) -> None:
    run_id = f"pg-{secrets.token_hex(4)}"

    ctx = make_ctx(FakeBrowser(SHOP), [PLAN, tool("Click", index=1)], database_url=pg_url)
    async with open_checkpointer(ctx.settings) as saver:
        state, pending, _ = await run_task(ctx, saver=saver, run_id=run_id)
    assert pending is not None and pending["kind"] == "approval"

    # New connection, new browser, new models: as if the process had been killed.
    browser = FakeBrowser(SHOP)
    ctx2 = make_ctx(browser, [tool("Done", result="resumed")], database_url=pg_url)
    async with open_checkpointer(ctx2.settings) as saver2:
        graph = compile_graph(saver2)
        snapshot = await graph.aget_state({"configurable": {"thread_id": run_id}})
        assert snapshot.next == ("human",)
        assert snapshot.values["plan"] == state["plan"]
        final, _ = await drive(graph, ctx2, run_id, None, max_steps=25, on_human=approve)

    assert final["status"] == "done" and final["result"] == "resumed"
    assert browser.executed[0].kind == "navigate"  # session re-created at the last URL
