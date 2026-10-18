from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent.context import AgentContext
from agent.llm import ask_text, usage_update
from agent.messages import fence_untrusted
from agent.prompts import load
from agent.routing import is_looping
from agent.state import AgentState, PendingTool
from agent.tools import AskHuman, ToolParseError, parse_agent_tool
from browser.actions import AnyAction, Done, Extract
from browser.models import ActionResult, clip_message, short_url
from browser.service import SessionNotFoundError

# A page that changes under every observation would otherwise loop observe→decide→act forever
# without spending a step.
MAX_STALE_IN_ROW = 2
_INDEX = re.compile(r"\[\d+\]")


def signature(pending: PendingTool, state: AgentState, result: ActionResult) -> str:
    """Identity of "the same action with the same outcome", independent of element numbering."""
    args = {k: v for k, v in pending["args"].items() if k != "index"}
    el = state["elements"].get(str(pending["args"].get("index")))
    target = el["name"] if el is not None else ""
    return "|".join(
        [
            pending["name"],
            str(target),
            json.dumps(args, sort_keys=True, ensure_ascii=False),
            state["current_url"] or "",
            _INDEX.sub("[]", result.message),
        ]
    )


async def act_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    pending = state["pending"]
    assert pending is not None, "act reached without a pending tool call"

    action: AnyAction | None = None
    try:
        tool = parse_agent_tool(pending["name"], pending["args"])
        if isinstance(tool, AskHuman | Done):
            raise ToolParseError(f"{pending['name']} is not a browser action")
        action = tool
    except ToolParseError as err:
        result = ActionResult(
            ok=False, message=clip_message(f"invalid call {pending['name']} → error: {err}")
        )
    else:
        try:
            result = await ctx.browser.execute(
                state["session_id"], action, expected_version=state["dom_version"]
            )
        except SessionNotFoundError:
            result = ActionResult(
                ok=False, stale=True, message="browser session was restarted — re-observing"
            )

    base: dict[str, Any] = {"pending": None, "human_approved": False}
    if result.stale and state["stale_in_row"] < MAX_STALE_IN_ROW:
        return {
            **base,
            "last_result": {"ok": False, "stale": True, "message": result.message},
            "stale_in_row": state["stale_in_row"] + 1,
        }

    update: dict[str, Any] = {}
    message, ok = result.message, result.ok
    if isinstance(action, Extract) and result.ok and result.extracted_text is not None:
        answer, tokens_in, tokens_out = await ask_text(
            ctx.llm_small,
            [
                SystemMessage(load("extract").text),
                HumanMessage(
                    f"Question: {action.query}\nPage URL: {result.url}\n\n"
                    + fence_untrusted(result.extracted_text)
                ),
            ],
        )
        update.update(usage_update(ctx, ctx.settings.model_small, tokens_in, tokens_out))
        answer = " ".join(answer.split())[:400]
        message = clip_message(f'extract "{action.query}" → {answer}')
        if answer.upper().startswith("NOT FOUND"):
            ok = False  # no progress: lets loop detection and reflection kick in
        else:
            update["findings"] = [f"{action.query} ({short_url(result.url)}): {answer}"]

    step = state["step"] + 1
    sig = signature(pending, state, result)
    signatures = [*state["signatures"], sig][-6:]
    progressed = ok and not is_looping(signatures)
    return {
        **base,
        **update,
        "step": step,
        "history": [f"{step}. {message}"],
        "last_result": {"ok": ok, "stale": False, "message": message, "url": result.url},
        "signatures": signatures,
        "errors_in_row": 0 if ok else state["errors_in_row"] + 1,
        "stale_in_row": 0,
        "reflections_in_row": 0 if progressed else state["reflections_in_row"],
        "notes": None if progressed else state["notes"],
        "current_url": result.url or state["current_url"],
    }
