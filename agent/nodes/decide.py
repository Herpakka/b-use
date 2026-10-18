from __future__ import annotations

import base64
from typing import Any

from langgraph.runtime import Runtime

from agent.context import AgentContext
from agent.llm import NoToolCallError, call_tool, usage_update
from agent.messages import build_decide_messages
from agent.policy import approval_reason
from agent.routing import limit_reason
from agent.state import AgentState, PendingTool
from agent.tools import ACTION_TOOL_SPECS
from browser.service import SessionNotFoundError

VISION_AFTER_REFLECTIONS = 2


async def decide_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """One LLM call → one pending tool call. Also applies the limits and the human gate."""
    ctx = runtime.context
    now = ctx.clock()
    if reason := limit_reason(state, now):
        return {"status": "failed", "failure_reason": reason, "pending": None}

    screenshot: str | None = None
    if state["vision"] or state["reflections_in_row"] >= VISION_AFTER_REFLECTIONS:
        try:
            screenshot = base64.b64encode(
                await ctx.browser.screenshot(state["session_id"])
            ).decode()
        except SessionNotFoundError:
            screenshot = None

    messages = build_decide_messages(state, ctx.settings, screenshot)
    usage: dict[str, Any] = {}
    try:
        call = await call_tool(ctx.llm_decide, messages, ACTION_TOOL_SPECS)
        pending = PendingTool(name=call.name, args=call.args)
        usage = usage_update(ctx, ctx.settings.model_decide, call.tokens_in, call.tokens_out)
    except NoToolCallError as err:
        # `act` turns this into an error step, so a model that keeps refusing still hits limits.
        pending = PendingTool(name="(no tool call)", args={"reply": str(err)})

    update: dict[str, Any] = {
        "pending": pending,
        "decided_at": now,
        "human_approved": False,
        "pending_reason": None,
        "status": "running",
        **usage,
    }
    if pending["name"] == "AskHuman":
        update.update(status="needs_human", pending_reason="the agent asked for help")
    elif reason := approval_reason(pending, state["elements"]):
        update.update(status="needs_human", pending_reason=reason)
    return update
