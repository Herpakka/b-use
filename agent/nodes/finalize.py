from __future__ import annotations

import contextlib
from typing import Any

from langgraph.runtime import Runtime
from pydantic import ValidationError

from agent.context import AgentContext
from agent.routing import MAX_REFLECTIONS_IN_ROW, limit_reason
from agent.state import AgentState
from browser.actions import Done


async def finalize_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    pending = state["pending"]
    update: dict[str, Any] = {"pending": None}

    if state["status"] != "failed" and pending is not None and pending["name"] == "Done":
        try:
            done = Done.model_validate(pending["args"])
        except ValidationError:
            done = Done(result=str(pending["args"].get("result", "")), success=False)
        step = state["step"] + 1
        update.update(
            step=step,
            status="done" if done.success else "failed",
            result=done.result,
            failure_reason=None if done.success else "the agent reported it could not finish",
            history=[f"{step}. done → {' '.join(done.result.split())[:150]}"],
        )
    elif state["status"] != "failed":
        reason = limit_reason(state, ctx.clock())
        if reason is None and state["reflections_in_row"] >= MAX_REFLECTIONS_IN_ROW:
            reason = f"stuck after {MAX_REFLECTIONS_IN_ROW} reflections in a row"
        update.update(status="failed", failure_reason=reason or "stopped")

    sid = state["session_id"]
    with contextlib.suppress(Exception):
        if own_session(sid, state["run_id"]):
            await ctx.browser.forget(sid)  # nobody can reuse it: don't keep its cookies
        else:
            await ctx.browser.close(sid)  # saved for the next run under this session name
    return update


def own_session(session_id: str, run_id: str) -> bool:
    """True for a run's own unnamed session (``run-…``, or ``<tenant>.run-…`` in the API)."""
    return session_id == run_id or session_id.endswith("." + run_id)
