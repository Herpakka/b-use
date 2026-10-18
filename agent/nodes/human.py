"""Pause for a person. The interrupt payload is what a UI renders (phase 3 debug page / CLI).

Payload:  {"kind": "approval" | "question", "question", "action", "reason", "url", "step"}
Resume:   {"approved": bool, "answer": str}   (a bare string or bool is accepted too)

Note: LangGraph re-runs this node from the top on resume, so nothing before ``interrupt()``
may have side effects.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from agent.context import AgentContext
from agent.state import AgentState, PendingTool


def describe_pending(pending: PendingTool, state: AgentState) -> str:
    args = pending["args"]
    target = state["elements"].get(str(args.get("index")))
    parts = [pending["name"]]
    if target is not None:
        parts.append(f'[{args["index"]}] "{target["name"]}"')
    parts += [f"{k}={v!r}" for k, v in args.items() if k != "index"]
    return " ".join(parts)


def parse_response(response: Any) -> tuple[bool, str]:
    if isinstance(response, dict):
        return bool(response.get("approved", False)), str(response.get("answer", "") or "")
    if isinstance(response, bool):
        return response, ""
    return True, str(response or "")


async def human_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    pending = state["pending"]
    assert pending is not None
    is_question = pending["name"] == "AskHuman"
    desc = describe_pending(pending, state)
    payload: dict[str, Any] = {
        "kind": "question" if is_question else "approval",
        "question": str(pending["args"].get("question", "")) if is_question else None,
        "action": None if is_question else desc,
        "reason": state["pending_reason"],
        "url": state["current_url"],
        "step": state["step"],
    }
    approved, answer = parse_response(interrupt(payload))

    # Waiting on a person must not eat into the run's wall-clock budget.
    waited = max(0.0, runtime.context.clock() - state["decided_at"])
    base: dict[str, Any] = {"status": "running", "started_at": state["started_at"] + waited}
    if is_question:
        return {
            **base,
            "pending": None,
            "human_approved": False,
            "notes": f"The human answered: {answer}",
            "history": [f'asked human "{str(payload["question"])[:80]}" → "{answer[:160]}"'],
        }
    if approved:
        return {**base, "human_approved": True, "history": [f"human approved: {desc}"]}
    note = f' — "{answer[:120]}"' if answer else ""
    return {
        **base,
        "pending": None,
        "human_approved": False,
        "notes": f"The human refused: {desc}{note}. Do not retry it; find another way or finish.",
        "history": [f"human denied: {desc}{note}"],
    }
