from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import ValidationError

from agent.context import AgentContext
from agent.llm import call_tool, usage_update
from agent.prompts import load
from agent.state import AgentState
from agent.tools import PLAN_TOOL_SPEC, Plan


async def plan_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Split the task into subgoals — once per run."""
    ctx = runtime.context
    request = "\n".join(
        [
            f"Task: {state['task']}",
            f"Allowed domains: {', '.join(state['allowed_domains'])}",
            f"Start URL: {state['start_url'] or '(none — the agent navigates itself)'}",
        ]
    )
    call = await call_tool(
        ctx.llm_decide,
        [SystemMessage(load("plan").text), HumanMessage(request)],
        [PLAN_TOOL_SPEC],
    )
    try:
        subgoals = Plan.model_validate(call.args).subgoals
    except ValidationError:
        subgoals = [state["task"]]
    return {
        "plan": subgoals,
        **usage_update(ctx, ctx.settings.model_decide, call.tokens_in, call.tokens_out),
    }
