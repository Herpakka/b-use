"""In-memory services for API tests: the real app, manager, SSE and WebSocket code paths, with
a fake browser, scripted LLM, memory store, local bus and in-memory checkpoints."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.context import AgentContext
from agent.graph import compile_graph
from api.main import create_app
from api.manager import RunManager
from api.services import Services, ServicesFactory
from core.bus import LocalBus
from core.config import Settings
from core.costs import PriceTable
from core.ratelimit import LocalRateLimiter
from core.secrets import MemorySecretStore
from core.store import MemoryRunStore
from tests.agent_fakes import FakeBrowser, ScriptedChatModel, page
from tests.conftest import make_settings

KEY_A = "acme-key-0123456789abcdef"
KEY_B = "globex-key-0123456789abcdef"
AUTH_A = {"Authorization": f"Bearer {KEY_A}"}
AUTH_B = {"Authorization": f"Bearer {KEY_B}"}

HOME = "https://shop.test/"
SHOP = {HOME: page(HOME, ("a", "Kettles"), ("button", "Place order"), body="Welcome")}


def api_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "api_keys": {KEY_A: "acme", KEY_B: "globex"},
        "event_bus": "local",
        "sse_heartbeat_s": 0.2,
        "rate_limit_per_minute": 10_000,
        "log_json": False,
        **overrides,
    }
    return make_settings(Path("unused-sessions"), **values)


def memory_factory(
    browser: FakeBrowser,
    decide: list[AIMessage],
    small: list[AIMessage] | None = None,
    *,
    store: MemoryRunStore | None = None,
    saver: InMemorySaver | None = None,
    llm_delay: float = 0.0,
    with_llm: bool = True,
    secrets: MemorySecretStore | None = None,
    bus: LocalBus | None = None,
) -> ServicesFactory:
    run_store = store or MemoryRunStore()
    secret_store = secrets or MemorySecretStore()
    checkpoints = saver or InMemorySaver()

    @asynccontextmanager
    async def factory(settings: Settings) -> AsyncIterator[Services]:
        event_bus = bus or LocalBus()
        ctx = (
            AgentContext(
                browser=browser,
                llm_decide=ScriptedChatModel(script=list(decide), delay=llm_delay),
                llm_small=ScriptedChatModel(script=list(small or [])),
                settings=settings,
                prices=PriceTable.from_settings(settings),
                secrets=secret_store.get,
            )
            if with_llm
            else None
        )
        manager = RunManager(
            store=run_store,
            bus=event_bus,
            browser=browser,
            graph=compile_graph(checkpoints),
            ctx=ctx,
            settings=settings,
            secrets=secret_store,
        )
        await manager.start()
        try:
            yield Services(
                settings=settings,
                store=run_store,
                bus=event_bus,
                limiter=LocalRateLimiter(),
                browser=browser,
                manager=manager,
                secrets=secret_store,
            )
        finally:
            await manager.shutdown()

    return factory


def client_for(factory: ServicesFactory, **settings: object) -> TestClient:
    return TestClient(create_app(api_settings(**settings), factory))


def start_run(client: TestClient, **body: Any) -> dict[str, Any]:
    payload = {"task": "find the answer", "allowed_domains": ["shop.test"], "start_url": HOME}
    res = client.post("/v1/runs", json={**payload, **body}, headers=AUTH_A)
    assert res.status_code == 202, res.text
    run: dict[str, Any] = res.json()
    return run


def wait_for(
    client: TestClient,
    run_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    timeout: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        run: dict[str, Any] = client.get(f"/v1/runs/{run_id}", headers=AUTH_A).json()
        if predicate(run):
            return run
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting; last state: {run}")
        time.sleep(0.05)


def status_in(*statuses: str) -> Callable[[dict[str, Any]], bool]:
    return lambda run: run["status"] in statuses


def event_types(client: TestClient, run_id: str) -> list[str]:
    return [e["type"] for e in client.get(f"/v1/runs/{run_id}/events", headers=AUTH_A).json()]


@contextmanager
def live_server(factory: ServicesFactory, **settings: object) -> Iterator[str]:
    """A real uvicorn server in a thread. TestClient buffers streaming responses until they end,
    so anything that reads SSE while the run is still going needs a real server."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    config = uvicorn.Config(
        create_app(api_settings(**settings), factory),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("test server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)
