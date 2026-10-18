"""The HTTP/SSE/WebSocket API over in-memory services (no Postgres, Redis, OpenAI or Chromium)."""

from __future__ import annotations

import time
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from starlette.websockets import WebSocketDisconnect

from browser.actions import Click
from bu_client import Client, Event, parse_sse
from core.store import MemoryRunStore
from tests.agent_fakes import PLAN, FakeBrowser, tool
from tests.api_support import (
    AUTH_A,
    AUTH_B,
    HOME,
    KEY_A,
    SHOP,
    client_for,
    event_types,
    live_server,
    memory_factory,
    start_run,
    status_in,
    wait_for,
)

DONE = tool("Done", result="7 kettles")


def happy_factory(browser: FakeBrowser | None = None, **kw: Any) -> Any:
    return memory_factory(browser or FakeBrowser(SHOP), [PLAN, tool("Click", index=1), DONE], **kw)


# ---------------------------------------------------------------- auth & limits
def test_requests_need_a_valid_key() -> None:
    with client_for(happy_factory()) as client:
        assert client.get("/v1/runs").status_code == 401
        assert client.get("/v1/runs", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.get("/v1/runs", headers={"X-API-Key": KEY_A}).status_code == 200
        assert client.get("/healthz").status_code == 200  # health stays open


def test_no_keys_configured_refuses_everything() -> None:
    with client_for(happy_factory(), api_keys={}) as client:
        assert client.get("/v1/runs", headers=AUTH_A).status_code == 503


def test_rate_limit_per_tenant() -> None:
    with client_for(happy_factory(), rate_limit_per_minute=3) as client:
        codes = [client.get("/v1/runs", headers=AUTH_A).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert client.get("/v1/runs", headers=AUTH_B).status_code == 200  # other tenant unaffected


def test_active_run_limit_and_budget_cap() -> None:
    factory = memory_factory(FakeBrowser(SHOP), [PLAN, DONE], llm_delay=0.5)
    with client_for(factory, max_active_runs_per_tenant=1) as client:
        start_run(client)
        res = client.post(
            "/v1/runs", json={"task": "t", "allowed_domains": ["shop.test"]}, headers=AUTH_A
        )
        assert res.status_code == 429
        res = client.post(
            "/v1/runs",
            json={"task": "t", "allowed_domains": ["shop.test"], "budget_usd": 50},
            headers=AUTH_B,
        )
        assert res.status_code == 422 and "budget_usd" in res.text


@pytest.mark.parametrize(
    "body",
    [
        {"task": "", "allowed_domains": ["a.test"]},
        {"task": "t", "allowed_domains": []},
        {"task": "t", "allowed_domains": ["  "]},
        {"task": "t", "allowed_domains": ["a.test"], "start_url": "file:///etc/passwd"},
        {"task": "t", "allowed_domains": ["a.test"], "session": "../other-tenant"},
        {"task": "t", "allowed_domains": ["a.test"], "max_steps": 0},
    ],
)
def test_create_run_validation(body: dict[str, Any]) -> None:
    with client_for(happy_factory()) as client:
        assert client.post("/v1/runs", json=body, headers=AUTH_A).status_code == 422


def test_without_openai_key_runs_are_refused() -> None:
    with client_for(happy_factory(with_llm=False)) as client:
        res = client.post(
            "/v1/runs", json={"task": "t", "allowed_domains": ["a.test"]}, headers=AUTH_A
        )
        assert res.status_code == 503 and "OpenAI" in res.text


# ---------------------------------------------------------------- run lifecycle
def test_run_to_completion_with_events_and_screenshot() -> None:
    browser = FakeBrowser(SHOP)
    with client_for(happy_factory(browser)) as client:
        res = client.post(
            "/v1/runs",
            json={"task": "count kettles", "allowed_domains": ["shop.test"], "start_url": HOME},
            headers=AUTH_A,
        )
        assert res.status_code == 202
        run = res.json()
        assert res.headers["location"] == f"/v1/runs/{run['id']}"
        assert run["status"] == "queued" and run["session"] == run["id"]

        with client.stream("GET", f"/v1/runs/{run['id']}/stream", headers=AUTH_A) as stream:
            events = list(parse_sse(stream.iter_lines()))
        assert [e.type for e in events] == [
            "created",
            "status",
            "plan",
            "observe",
            "decision",
            "step",
            "observe",
            "decision",
            "finished",
        ]
        assert [e.id for e in events] == list(range(1, 10))
        assert events[4].data["tool"] == "Click" and events[5].data["line"] == "1. click → ok"
        assert events[-1].data["status"] == "done"

        final = client.get(f"/v1/runs/{run['id']}", headers=AUTH_A).json()
        assert final["status"] == "done" and final["result"] == "7 kettles"
        assert final["steps"] == 2 and final["cost_usd"] > 0 and final["finished_at"]

        shot = client.get(f"/v1/runs/{run['id']}/screenshot", headers=AUTH_A)
        assert shot.status_code == 200 and shot.headers["content-type"] == "image/jpeg"
        assert shot.content.startswith(b"\xff\xd8\xff")  # stored after the step

        listed = client.get("/v1/runs", headers=AUTH_A).json()
        assert [r["id"] for r in listed] == [run["id"]]


def test_sse_resumes_after_last_event_id() -> None:
    with client_for(happy_factory()) as client:
        run = start_run(client)
        wait_for(client, run["id"], status_in("done"))
        with client.stream(
            "GET", f"/v1/runs/{run['id']}/stream", headers={**AUTH_A, "Last-Event-ID": "6"}
        ) as res:
            assert res.headers["content-type"].startswith("text/event-stream")
            assert res.headers["x-accel-buffering"] == "no"
            ids = [e.id for e in parse_sse(res.iter_lines())]
        assert ids == [7, 8, 9]


def test_approval_flow_over_the_api() -> None:
    browser = FakeBrowser(SHOP)
    factory = memory_factory(browser, [PLAN, tool("Click", index=2), tool("Done", result="ok")])
    with client_for(factory) as client:
        run = start_run(client)
        paused = wait_for(client, run["id"], status_in("needs_human"))
        assert paused["pending"]["kind"] == "approval"
        assert paused["pending"]["action"] == 'Click [2] "Place order"'
        assert Click(index=2) not in browser.executed

        assert (
            client.post(f"/v1/runs/{run['id']}/resume", json={}, headers=AUTH_B).status_code == 404
        )
        res = client.post(f"/v1/runs/{run['id']}/resume", json={"approved": True}, headers=AUTH_A)
        assert res.status_code == 202
        done = wait_for(client, run["id"], status_in("done"))
        assert done["pending"] is None
        assert Click(index=2) in browser.executed
        again = client.post(f"/v1/runs/{run['id']}/resume", json={}, headers=AUTH_A)
        assert again.status_code == 409
        assert "resumed" in event_types(client, run["id"])


def test_sdk_wait_answers_the_human() -> None:
    factory = memory_factory(
        FakeBrowser(SHOP), [PLAN, tool("AskHuman", question="Which size?"), DONE]
    )
    asked: list[dict[str, Any]] = []
    seen: list[Event] = []

    def on_human(pending: dict[str, Any]) -> dict[str, Any]:
        asked.append(pending)
        return {"approved": True, "answer": "Large"}

    with live_server(factory) as base_url, Client(base_url, api_key=KEY_A) as sdk:
        run = sdk.create_run("buy a kettle", ["shop.test"], start_url=HOME)
        final = sdk.wait(run["id"], on_event=seen.append, on_human=on_human)
    assert final["status"] == "done"
    assert asked[0]["kind"] == "question" and asked[0]["question"] == "Which size?"
    assert "human" in [e.type for e in seen]


def test_cancel_a_running_run() -> None:
    browser = FakeBrowser(SHOP)
    factory = memory_factory(browser, [PLAN, DONE], llm_delay=0.4)
    with client_for(factory) as client:
        run = start_run(client)
        wait_for(client, run["id"], status_in("running"))
        res = client.post(f"/v1/runs/{run['id']}/cancel", headers=AUTH_A)
        assert res.status_code == 200 and res.json()["status"] == "cancelled"
        assert client.post(f"/v1/runs/{run['id']}/cancel", headers=AUTH_A).status_code == 409
        assert event_types(client, run["id"])[-1] == "finished"
        time.sleep(0.5)  # the cancelled task must not write anything afterwards
        assert client.get(f"/v1/runs/{run['id']}", headers=AUTH_A).json()["status"] == "cancelled"


def test_tenants_are_isolated() -> None:
    with client_for(happy_factory()) as client:
        run = start_run(client, session="shop-login")
        wait_for(client, run["id"], status_in("done"))
        assert run["session"] == "shop-login"
        assert client.get(f"/v1/runs/{run['id']}", headers=AUTH_B).status_code == 404
        assert client.get(f"/v1/runs/{run['id']}/events", headers=AUTH_B).status_code == 404
        assert client.get(f"/v1/runs/{run['id']}/screenshot", headers=AUTH_B).status_code == 404
        assert client.get("/v1/runs", headers=AUTH_B).json() == []


def test_named_sessions_are_namespaced_per_tenant() -> None:
    browser = FakeBrowser(SHOP)
    factory = memory_factory(browser, [PLAN, tool("Click", index=2)])
    with client_for(factory) as client:
        run = start_run(client, session="shop-login")
        wait_for(client, run["id"], status_in("needs_human"))
        assert browser.sessions == {"acme.shop-login"}
        mine = client.get("/v1/sessions", headers=AUTH_A).json()
        assert [s["name"] for s in mine] == ["shop-login"]
        assert client.get("/v1/sessions", headers=AUTH_B).json() == []
        assert client.delete("/v1/sessions/shop-login", headers=AUTH_B).status_code == 404
        # a second active run may not grab the same session
        res = client.post(
            "/v1/runs",
            json={"task": "t", "allowed_domains": ["shop.test"], "session": "shop-login"},
            headers=AUTH_A,
        )
        assert res.status_code == 409
        assert client.delete("/v1/sessions/shop-login", headers=AUTH_A).status_code == 204
        assert browser.sessions == set()


def test_live_view_websocket() -> None:
    factory = memory_factory(FakeBrowser(SHOP), [PLAN, tool("Click", index=2)])
    with client_for(factory) as client:
        run = start_run(client)
        wait_for(client, run["id"], status_in("needs_human"))  # session stays open while paused
        with client.websocket_connect(
            f"/v1/runs/{run['id']}/live", subprotocols=[f"bu.key.{KEY_A}"]
        ) as ws:
            assert ws.accepted_subprotocol == f"bu.key.{KEY_A}"
            frames = [ws.receive_bytes() for _ in range(3)]
            assert all(f.startswith(b"\xff\xd8\xff") for f in frames)
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_bytes()  # the fake screencast ends → server closes
            assert closed.value.code == 4410
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(f"/v1/runs/{run['id']}/live"),
        ):
            pass  # no key → refused


def test_running_run_recovers_after_restart() -> None:
    store, saver = MemoryRunStore(), InMemorySaver()
    browser = FakeBrowser(SHOP)
    slow = memory_factory(browser, [PLAN, DONE], store=store, saver=saver, llm_delay=1.0)
    with client_for(slow) as client:
        run = start_run(client)
        # plan takes 1 s; leave once it is done and the next LLM call is in flight
        wait_for(client, run["id"], lambda r: "observe" in event_types(client, r["id"]))
    assert store.runs[run["id"]].status == "running"  # shutdown leaves it resumable

    fresh = memory_factory(FakeBrowser(SHOP), [DONE], store=store, saver=saver)
    with client_for(fresh) as client:
        final = wait_for(client, run["id"], status_in("done", "failed"))
        assert final["status"] == "done", final
        assert "recovered" in event_types(client, run["id"])


def test_debug_page_and_root_redirect() -> None:
    with client_for(happy_factory()) as client:
        page_res = client.get("/debug")
        assert page_res.status_code == 200 and "Browser Agent" in page_res.text
        assert "default-src 'self'" in page_res.headers["content-security-policy"]
        assert client.get("/", follow_redirects=False).headers["location"] == "/debug"
    with client_for(happy_factory(), debug_page=False) as client:
        assert client.get("/debug").status_code == 404


def test_openapi_lists_the_public_routes() -> None:
    with client_for(happy_factory()) as client:
        paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/v1/runs",
        "/v1/runs/{run_id}",
        "/v1/runs/{run_id}/stream",
        "/v1/runs/{run_id}/resume",
        "/v1/runs/{run_id}/cancel",
        "/v1/runs/{run_id}/screenshot",
        "/v1/runs/{run_id}/events",
        "/v1/sessions",
        "/v1/sessions/{name}",
    } <= paths


def test_password_text_is_masked_in_events() -> None:
    from tests.agent_fakes import page

    login = {HOME: page(HOME, ("input", "Username", "text"), ("input", "Password", "password"))}
    script = [
        PLAN,
        tool("TypeText", index=1, text="tomsmith"),
        tool("TypeText", index=2, text="hunter2-secret"),
        DONE,
    ]
    with client_for(memory_factory(FakeBrowser(login), script)) as client:
        run = start_run(client)
        wait_for(client, run["id"], status_in("done"))
        events = client.get(f"/v1/runs/{run['id']}/events", headers=AUTH_A).text
    assert "hunter2-secret" not in events
    assert "tomsmith" in events  # only password fields are masked
