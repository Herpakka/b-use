"""5.3: a run's trace reaches Langfuse — one session per run, a span per node, PII masked.

Needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST (e.g. the local Langfuse:
http://host.docker.internal:7453 from the container). Skipped otherwise.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent.tracing import Tracer
from core.config import get_settings
from tests.agent_fakes import PLAN, FakeBrowser, make_ctx, page, run_task, tool

pytestmark = pytest.mark.langfuse

HOME = "https://shop.test/"


async def test_run_is_traced_with_masked_pii() -> None:
    env = get_settings()
    if not (env.langfuse_public_key and env.langfuse_secret_key):
        pytest.skip("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
    tracer = Tracer(env)
    browser = FakeBrowser({HOME: page(HOME, ("a", "Kettles"), body="Contact alice@example.com")})
    ctx = replace(make_ctx(browser, [PLAN, tool("Done", result="ok")]), tracer=tracer)
    final, _, _ = await run_task(ctx, task="find the answer for alice@example.com")
    run_id = final["run_id"]
    tracer.flush()

    api = tracer.client.api.observations
    roots = []
    for _ in range(60):  # ingestion is asynchronous (worker → ClickHouse)
        # The session is set on the trace's root span (one trace per graph invocation).
        roots = api.get_many(session_id=run_id).data
        if roots:
            break
        await asyncio.sleep(1)
    assert roots, f"no trace for session {run_id}"
    observations = api.get_many(trace_id=roots[0].trace_id, limit=100, fields="core,basic,io").data
    tracer.shutdown()
    names = {o.name for o in observations}
    assert {"plan", "observe", "decide", "finalize"} <= names, names
    dumped = " ".join(str(o.input) + str(o.output) for o in observations)
    assert "alice@example.com" not in dumped and "[email]" in dumped
