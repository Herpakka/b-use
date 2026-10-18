"""Two replicas sharing one store, bus and checkpointer — as two app containers sharing Postgres
and Redis — each with its own browser. Leases are short so ownership changes hands quickly."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.context import AgentContext
from agent.graph import compile_graph
from api.manager import RunError, RunManager
from api.schemas import CreateRunRequest, ResumeRequest
from browser.actions import Click
from core.bus import LocalBus
from core.costs import PriceTable
from core.store import MemoryRunStore, RunRecord
from tests.agent_fakes import PLAN, FakeBrowser, ScriptedChatModel, tool
from tests.api_support import HOME, SHOP, api_settings

DONE = tool("Done", result="ok")


class Cluster:
    def __init__(self) -> None:
        self.store = MemoryRunStore()
        self.bus = LocalBus()
        self.saver = InMemorySaver()
        self.browsers: dict[str, FakeBrowser] = {}

    def replica(
        self, name: str, script: list[AIMessage], *, delay: float = 0.0, **settings: Any
    ) -> RunManager:
        s = api_settings(
            **{"replica_id": name, "run_lease_s": 0.6, "drain_timeout_s": 5, **settings}
        )
        browser = self.browsers[name] = FakeBrowser(SHOP)
        ctx = AgentContext(
            browser=browser,
            llm_decide=ScriptedChatModel(script=list(script), delay=delay),
            llm_small=ScriptedChatModel(),
            settings=s,
            prices=PriceTable.from_settings(s),
        )
        return RunManager(
            store=self.store,
            bus=self.bus,
            browser=browser,
            graph=compile_graph(self.saver),
            ctx=ctx,
            settings=s,
        )

    async def until(
        self, run_id: str, predicate: Callable[[RunRecord], bool], within_s: float = 10
    ) -> RunRecord:
        deadline = asyncio.get_running_loop().time() + within_s
        while True:
            run = await self.store.get_run(run_id)
            assert run is not None
            if predicate(run):
                return run
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(f"timed out; run is {run.status}, owner {run.owner}")
            await asyncio.sleep(0.05)

    def events(self, run_id: str) -> list[str]:
        return [e.type for e in self.store.events.get(run_id, [])]


def request(**kw: Any) -> CreateRunRequest:
    return CreateRunRequest.model_validate(
        {"task": "find it", "allowed_domains": ["shop.test"], "start_url": HOME, **kw}
    )


def status(*names: str) -> Callable[[RunRecord], bool]:
    return lambda run: run.status in names


async def test_resume_on_another_replica_runs_on_the_owner() -> None:
    c = Cluster()
    a = c.replica("a", [PLAN, tool("Click", index=2), DONE])  # "Place order" → approval
    b = c.replica("b", [])  # must never be asked anything
    await a.start()
    await b.start()
    try:
        run = await a.create("acme", request())
        paused = await c.until(run.id, status("needs_human"))
        assert paused.owner == "a"

        await b.resume("acme", run.id, ResumeRequest(approved=True))
        done = await c.until(run.id, status("done", "failed"))
        assert done.status == "done" and done.owner is None
        assert Click(index=2) in c.browsers["a"].executed  # where the paused session lived
        assert c.browsers["b"].executed == []
    finally:
        await a.shutdown()
        await b.shutdown()


async def test_cancel_on_another_replica_stops_the_owner() -> None:
    c = Cluster()
    a = c.replica("a", [PLAN] + [tool("Scroll", direction="down")] * 20, delay=0.3)
    b = c.replica("b", [])
    await a.start()
    await b.start()
    try:
        run = await a.create("acme", request())
        await c.until(run.id, lambda r: "observe" in c.events(r.id))
        cancelled = await b.cancel("acme", run.id)
        assert cancelled.status == "cancelled"
        await asyncio.sleep(0.1)
        assert not a.is_running(run.id)
    finally:
        await a.shutdown()
        await b.shutdown()


async def test_runs_of_a_dead_replica_are_adopted() -> None:
    c = Cluster()
    a = c.replica("a", [PLAN, DONE], delay=1.0)
    b = c.replica("b", [tool("Done", result="finished by b")])
    await a.start()
    run = await a.create("acme", request())
    await c.until(run.id, lambda r: "observe" in c.events(r.id))
    # "a" dies: no drain, no lease release — its leases just stop being renewed.
    for task in [*a._tasks.values(), *a._loops]:
        task.cancel()
    await asyncio.gather(*a._tasks.values(), *a._loops, return_exceptions=True)
    assert (await c.until(run.id, status("running"))).owner == "a"

    await b.start()
    try:
        done = await c.until(run.id, status("done", "failed"))
        assert done.status == "done" and done.result == "finished by b"
        recovered = [e for e in c.store.events[run.id] if e.type == "recovered"]
        assert recovered and recovered[0].data["by"] == "b"
    finally:
        await b.shutdown()


async def test_shutdown_drains_and_hands_over_the_rest() -> None:
    c = Cluster()
    quick = c.replica("a", [PLAN, DONE])
    await quick.start()
    finished = await quick.create("acme", request())
    await quick.shutdown()  # waits for the run instead of abandoning it
    assert (await c.store.get_run(finished.id)).status == "done"  # type: ignore[union-attr]
    with pytest.raises(RunError) as refused:
        await quick.create("acme", request())
    assert refused.value.status == 503

    slow = c.replica("a2", [PLAN, DONE], delay=1.0, drain_timeout_s=0.2)
    b = c.replica("b", [tool("Done", result="finished by b")])
    await slow.start()
    await b.start()
    try:
        run = await slow.create("acme", request())
        await c.until(run.id, lambda r: "observe" in c.events(r.id))
        await slow.shutdown()  # drain times out: the run is stopped and its lease released
        assert (await c.store.get_run(run.id)).owner != "a2"  # type: ignore[union-attr]
        done = await c.until(run.id, status("done", "failed"), within_s=5)
        assert done.result == "finished by b"
    finally:
        await b.shutdown()


async def test_admission_is_one_at_a_time_across_replicas() -> None:
    c = Cluster()
    a = c.replica("a", [PLAN, DONE], delay=0.5, max_active_runs_per_tenant=1)
    b = c.replica("b", [PLAN, DONE], delay=0.5, max_active_runs_per_tenant=1)
    await a.start()
    await b.start()
    try:
        results = await asyncio.gather(
            a.create("acme", request()), b.create("acme", request()), return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, RunError)]
        assert len(errors) == 1 and errors[0].status == 429
    finally:
        await a.shutdown()
        await b.shutdown()


async def test_a_full_replica_refuses_with_retry_after() -> None:
    c = Cluster()
    a = c.replica("a", [PLAN, tool("Click", index=2), DONE] * 2)
    c.browsers["a"].capacity = 1
    await a.start()
    try:
        run = await a.create("acme", request())  # pauses for approval, holding its browser
        await c.until(run.id, status("needs_human"))
        with pytest.raises(RunError) as full:
            await a.create("acme", request())
        assert full.value.status == 503 and full.value.retry_after == 5
    finally:
        await a.shutdown()
