"""Drive a compiled graph to completion, surfacing events and human interrupts.

Shared by the CLI and the eval runner; the API (phase 3) will use the same loop with an SSE
event sink and a resume endpoint instead of ``on_human``.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.context import AgentContext
from agent.graph import AgentGraph, run_config
from agent.prompts import prompt_version
from agent.state import AgentState, Limits, new_state
from core.config import Settings

EventSink = Callable[[str, dict[str, Any]], Awaitable[None] | None]
HumanHandler = Callable[[dict[str, Any]], Awaitable[Any]]


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def limits_from(
    settings: Settings,
    *,
    max_steps: int | None = None,
    budget_usd: float | None = None,
    run_timeout_s: float | None = None,
) -> Limits:
    return Limits(
        max_steps=max_steps or settings.max_steps,
        budget_usd=budget_usd or settings.budget_usd,
        run_timeout_s=run_timeout_s or settings.run_timeout_s,
    )


def initial_state(
    ctx: AgentContext,
    *,
    task: str,
    allowed_domains: list[str],
    run_id: str | None = None,
    session_id: str | None = None,
    start_url: str | None = None,
    vision: bool = False,
    limits: Limits | None = None,
) -> AgentState:
    return new_state(
        run_id=run_id or new_run_id(),
        task=task,
        allowed_domains=allowed_domains,
        limits=limits or limits_from(ctx.settings),
        prompt_version=prompt_version(),
        now=ctx.clock(),
        session_id=session_id,
        start_url=start_url,
        vision=vision,
    )


@asynccontextmanager
async def open_checkpointer(
    settings: Settings, *, memory: bool = False
) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    if memory:
        yield InMemorySaver()
        return
    # Imported lazily: psycopg's async mode cannot run on the Windows Proactor loop.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        await saver.setup()
        yield saver


async def drive(
    graph: AgentGraph,
    ctx: AgentContext,
    run_id: str,
    payload: AgentState | Command[Any] | None,
    *,
    max_steps: int,
    on_event: EventSink | None = None,
    on_human: HumanHandler | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run until the graph ends or pauses.

    Returns ``(state, pending_interrupt)``; the interrupt is None unless the run is paused
    and no ``on_human`` handler was given. ``payload=None`` resumes from the last checkpoint.
    """
    config = run_config(run_id, max_steps)
    stream_input: Any = payload
    while True:
        async for chunk in graph.astream(
            stream_input,
            config,
            context=ctx,
            stream_mode="updates",
            # Persist each checkpoint before the next step so a killed process resumes cleanly.
            durability="sync",
        ):
            for node, update in chunk.items():
                if node != "__interrupt__" and on_event is not None:
                    result = on_event(node, update or {})
                    if inspect.isawaitable(result):
                        await result
        snapshot = await graph.aget_state(config)
        if snapshot.interrupts:
            value = snapshot.interrupts[0].value
            if on_human is None:
                return dict(snapshot.values), value
            stream_input = Command(resume=await on_human(value))
            continue
        return dict(snapshot.values), None
