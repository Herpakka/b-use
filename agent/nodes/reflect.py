from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import ValidationError

from agent.context import AgentContext
from agent.llm import call_tool, usage_update
from agent.messages import fence_untrusted, history_block, numbered
from agent.prompts import load
from agent.state import AgentState
from agent.tools import REFLECT_TOOL_SPEC, Reflection

PAGE_CHARS = 4000


async def reflect_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Called after an error or a 3x repeated action: diagnose and change course."""
    ctx = runtime.context
    last = (state["last_result"] or {}).get("message", "")
    request = "\n\n".join(
        [
            f"# Task\n{state['task']}",
            f"# Plan\n{numbered(state['plan'])}",
            f"# Recent steps\n{history_block(state['history'], 10)}",
            f"# Last result\n{last}",
            "# Current page (untrusted)\n"
            + fence_untrusted((state["observation"] or "")[:PAGE_CHARS]),
        ]
    )
    call = await call_tool(
        ctx.llm_decide,
        [SystemMessage(load("reflect").text), HumanMessage(request)],
        [REFLECT_TOOL_SPEC],
    )
    try:
        r = Reflection.model_validate(call.args)
    except ValidationError:
        r = Reflection(diagnosis="(unreadable reflection)", revised_plan=state["plan"], advice="")
    diagnosis = " ".join(r.diagnosis.split())[:160]
    return {
        "plan": r.revised_plan or state["plan"],
        "notes": r.advice or None,
        "reflections_in_row": state["reflections_in_row"] + 1,
        "signatures": [],  # give the new approach a clean slate for loop detection
        "errors_in_row": 0,
        "history": [f"↺ reflected: {diagnosis}"],
        **usage_update(
            ctx, ctx.settings.model_decide, call.tokens_in, call.tokens_out, node="reflect"
        ),
    }
