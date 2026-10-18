"""Conditional edges and the run limits. Pure functions of state — easy to unit test."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from agent.state import AgentState

LOOP_REPEATS = 3
LOOP_WINDOW = 6
MAX_REFLECTIONS_IN_ROW = 4


def limit_reason(state: AgentState, now: float | None = None) -> str | None:
    """The three loop guards: step count, USD budget, wall-clock timeout."""
    limits = state["limits"]
    if state["step"] >= limits["max_steps"]:
        return f"max_steps ({limits['max_steps']}) reached"
    if state["cost_usd"] >= limits["budget_usd"]:
        return f"budget ${limits['budget_usd']:.2f} exhausted (${state['cost_usd']:.4f})"
    if now is not None and now - state["started_at"] >= limits["run_timeout_s"]:
        return f"timeout ({limits['run_timeout_s']:.0f}s) reached"
    return None


def is_looping(signatures: list[str]) -> bool:
    """The same action with the same outcome 3x within the last 6 steps — catches A,A,A and
    alternating cycles like A,B,A,B,A (e.g. select → check → select → check …)."""
    window = signatures[-LOOP_WINDOW:]
    return bool(window) and max(Counter(window).values()) >= LOOP_REPEATS


def route_after_decide(state: AgentState) -> Literal["human", "act", "finalize"]:
    if state["status"] == "failed" or state["pending"] is None:
        return "finalize"
    if state["pending"]["name"] == "Done":
        return "finalize"
    if state["status"] == "needs_human":
        return "human"
    return "act"


def route_after_human(state: AgentState) -> Literal["act", "observe"]:
    return "act" if state["human_approved"] else "observe"


def route_after_act(state: AgentState) -> Literal["observe", "reflect", "finalize"]:
    if limit_reason(state) is not None:
        return "finalize"
    last = state["last_result"] or {}
    if last.get("stale"):
        return "observe"
    if state["errors_in_row"] > 0 or is_looping(state["signatures"]):
        if state["reflections_in_row"] >= MAX_REFLECTIONS_IN_ROW:
            return "finalize"
        return "reflect"
    return "observe"
