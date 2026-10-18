"""Monthly spend caps: a run starts only if its budget fits beside what is already committed."""

from datetime import timedelta

from api.manager import month_start
from core.store import MemoryRunStore, RunRecord, utcnow
from tests.agent_fakes import PLAN, FakeBrowser, tool
from tests.api_support import AUTH_A, AUTH_B, SHOP, client_for, memory_factory, start_run

LIVE = utcnow() + timedelta(hours=1)


def past_run(store: MemoryRunStore, tenant: str, status: str, cost: float, **kw: object) -> None:
    run = RunRecord.model_validate(
        {
            "id": f"run-{len(store.runs):012d}",
            "tenant": tenant,
            "task": "earlier",
            "allowed_domains": ["shop.test"],
            "session_id": f"{tenant}.s",
            "max_steps": 5,
            "status": status,
            "cost_usd": cost,
            **kw,
        }
    )
    store.runs[run.id] = run


def test_tenant_cap_shrinks_then_refuses_runs() -> None:
    store = MemoryRunStore()
    past_run(store, "acme", "done", 0.10)
    # paused on a live replica, reserved at its full budget
    past_run(store, "acme", "needs_human", 0.01, budget_usd=0.05, owner="other", lease_until=LIVE)
    past_run(store, "acme", "done", 5.0, created_at=month_start() - timedelta(days=1))  # last month
    past_run(store, "globex", "done", 0.05)  # counted against globex only
    factory = memory_factory(FakeBrowser(SHOP), [PLAN, tool("Done", result="ok")], store=store)
    with client_for(factory, tenant_monthly_budget_usd=0.20) as client:
        usage = client.get("/v1/usage", headers=AUTH_A).json()
        assert usage["month"] == utcnow().strftime("%Y-%m")
        assert usage["runs"] == 2 and usage["cost_usd"] == 0.11
        assert usage["committed_usd"] == 0.15 and usage["remaining_usd"] == 0.05

        run = start_run(client)  # default budget 0.15 is cut to the 0.05 left
        assert run["budget_usd"] == 0.05

        res = client.post(
            "/v1/runs",
            json={"task": "x", "allowed_domains": ["shop.test"]},
            headers=AUTH_A,
        )
        assert res.status_code == 402 and "monthly budget exhausted" in res.text
        assert client.get("/v1/usage", headers=AUTH_B).json()["remaining_usd"] == 0.15


def test_deployment_cap_counts_every_tenant() -> None:
    store = MemoryRunStore()
    past_run(store, "globex", "done", 0.95)
    factory = memory_factory(FakeBrowser(SHOP), [PLAN, tool("Done", result="ok")], store=store)
    with client_for(factory, monthly_budget_usd=1.0) as client:
        assert client.get("/v1/usage", headers=AUTH_A).json()["remaining_usd"] == 0.05
        assert start_run(client)["budget_usd"] == 0.05


def test_no_caps_by_default() -> None:
    factory = memory_factory(FakeBrowser(SHOP), [PLAN, tool("Done", result="ok")])
    with client_for(factory) as client:
        usage = client.get("/v1/usage", headers=AUTH_A).json()
        assert usage["remaining_usd"] is None and usage["budget_usd"] is None
        assert start_run(client, budget_usd=0.5)["budget_usd"] == 0.5
