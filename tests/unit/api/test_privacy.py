"""4.8: PII stays out of the event log, and nothing is kept longer than its retention."""

from datetime import timedelta
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from api.janitor import Janitor
from bu_client import Client
from core.store import MemoryRunStore, RunRecord, utcnow
from tests.agent_fakes import PLAN, FakeBrowser, make_ctx, run_task, tool
from tests.api_support import (
    AUTH_A,
    HOME,
    KEY_A,
    SHOP,
    client_for,
    live_server,
    memory_factory,
    start_run,
    status_in,
    wait_for,
)
from tests.conftest import make_settings

EMAIL = "alice@example.com"


def test_events_are_masked_but_the_run_record_is_not() -> None:
    script = [
        PLAN,
        tool("AskHuman", question=f"Send the receipt to {EMAIL}?"),
        tool("Done", result=f"Receipt sent to {EMAIL}"),
    ]
    factory = memory_factory(FakeBrowser(SHOP), script)
    with client_for(factory) as client:
        run = start_run(client, task=f"Order a kettle for {EMAIL}, phone 081-234-5678")
        waiting = wait_for(client, run["id"], status_in("needs_human"))
        assert EMAIL in waiting["pending"]["question"]  # the human sees the real question
        client.post(f"/v1/runs/{run['id']}/resume", json={"answer": "yes"}, headers=AUTH_A)
        final = wait_for(client, run["id"], status_in("done", "failed"))

        assert EMAIL in final["task"] and EMAIL in final["result"]
        events = client.get(f"/v1/runs/{run['id']}/events", headers=AUTH_A).text
        assert EMAIL not in events and "081-234-5678" not in events
        assert "[email]" in events and "[phone]" in events


def test_sdk_hands_the_human_the_unmasked_question() -> None:
    script = [PLAN, tool("AskHuman", question=f"Use {EMAIL}?"), tool("Done", result="ok")]
    seen: list[dict[str, Any]] = []

    def answer(pending: dict[str, Any]) -> dict[str, Any]:
        seen.append(pending)
        return {"approved": True, "answer": "yes"}

    with live_server(memory_factory(FakeBrowser(SHOP), script)) as url, Client(url, KEY_A) as sdk:
        run = sdk.create_run("task", ["shop.test"], start_url=HOME)
        assert sdk.wait(run["id"], on_human=answer)["status"] == "done"
    assert seen and EMAIL in seen[0]["question"]


def test_redaction_can_be_switched_off() -> None:
    factory = memory_factory(FakeBrowser(SHOP), [PLAN, tool("Done", result="ok")])
    with client_for(factory, redact_pii=False) as client:
        run = start_run(client, task=f"mail {EMAIL}")
        wait_for(client, run["id"], status_in("done"))
        assert EMAIL in client.get(f"/v1/runs/{run['id']}/events", headers=AUTH_A).text


async def test_janitor_applies_every_retention_rule() -> None:
    saver = InMemorySaver()
    browser = FakeBrowser(SHOP)
    ctx = make_ctx(browser, [PLAN, tool("Done", result="ok")])
    final, _, graph = await run_task(ctx, saver=saver, run_id="run-000000000001")
    config: Any = {"configurable": {"thread_id": "run-000000000001"}}
    assert (await graph.aget_state(config)).values  # a checkpoint exists

    now = utcnow()
    store = MemoryRunStore()

    def add(run_id: str, status: str, ended_days_ago: float | None) -> None:
        ended = None if ended_days_ago is None else now - timedelta(days=ended_days_ago)
        store.runs[run_id] = RunRecord.model_validate(
            {
                "id": run_id,
                "tenant": "acme",
                "task": "t",
                "allowed_domains": ["shop.test"],
                "session_id": f"acme.{run_id}",
                "max_steps": 5,
                "status": status,
                "finished_at": ended,
            }
        )
        store.events[run_id] = []

    add("run-000000000001", "done", 2)  # finished two days ago: checkpoint goes
    add("run-000000000002", "running", None)  # active: untouched
    add("run-000000000003", "failed", 40)  # past run retention: deleted entirely
    await store.save_artifact("run-000000000002", "screenshot", b"new", "image/jpeg")
    await store.save_artifact("run-000000000001", "screenshot", b"old", "image/jpeg")
    old = store.artifacts[("run-000000000001", "screenshot")]
    store.artifacts[("run-000000000001", "screenshot")] = (old[0], old[1], now - timedelta(days=2))

    settings = make_settings(Path("unused"), run_retention_days=30)
    janitor = Janitor(store, saver, settings)
    done = await janitor.sweep(now)

    assert done == {"screenshots": 1, "checkpoints": 2, "runs": 1}
    assert not (await graph.aget_state(config)).values
    assert store.runs["run-000000000001"].purged_at == now
    assert await store.get_artifact("run-000000000002", "screenshot") is not None
    assert await store.get_artifact("run-000000000001", "screenshot") is None
    assert "run-000000000003" not in store.runs and "run-000000000003" not in store.events
    assert "run-000000000002" in store.runs
    # Idempotent: nothing left to do.
    assert await janitor.sweep(now) == {"screenshots": 0, "checkpoints": 0, "runs": 0}
    assert final["status"] == "done"
