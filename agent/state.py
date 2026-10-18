"""Graph state. Everything here is checkpointed, so it must stay JSON-like — no browser objects.

Context strategy: only the *latest* observation is kept; earlier steps survive as one-line
history entries, which keeps the prompt size flat however long the run gets.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

RunStatus = Literal["running", "needs_human", "done", "failed"]


def add_costs(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a.get(k, 0.0) + b.get(k, 0.0) for k in a.keys() | b.keys()}


class PendingTool(TypedDict):
    name: str
    args: dict[str, Any]


class ElementBrief(TypedDict):
    tag: str
    name: str
    type: str | None
    role: str | None
    # Missing in checkpoints from before phase 4.
    field: NotRequired[str | None]  # "payment" | "password"
    form: NotRequired[str | None]  # "payment" | "login" | "search" | "form"


class Limits(TypedDict):
    max_steps: int
    budget_usd: float
    run_timeout_s: float


class AgentState(TypedDict):
    run_id: str
    task: str
    allowed_domains: list[str]
    session_id: str
    start_url: str | None
    vision: bool
    limits: Limits
    prompt_version: str
    # Whose secrets {{SECRET:name}} placeholders resolve to (the API's tenant; "" locally).
    # Read with .get(): checkpoints written before phase 4 lack it.
    secret_scope: str

    plan: list[str]
    notes: str | None  # advice from the last reflection or the human

    observation: str | None
    dom_version: str | None
    current_url: str | None
    elements: dict[str, ElementBrief]  # str(index) → brief, for the approval policy

    history: Annotated[list[str], operator.add]
    findings: Annotated[list[str], operator.add]
    step: int
    status: RunStatus

    pending: PendingTool | None
    pending_reason: str | None
    human_approved: bool
    last_result: dict[str, Any] | None

    signatures: list[str]
    errors_in_row: int
    stale_in_row: int
    reflections_in_row: int

    cost_usd: Annotated[float, operator.add]
    node_costs: Annotated[dict[str, float], add_costs]  # plan/decide/reflect/extract
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    started_at: float
    decided_at: float

    result: str | None
    failure_reason: str | None


def new_state(
    *,
    run_id: str,
    task: str,
    allowed_domains: list[str],
    limits: Limits,
    prompt_version: str,
    now: float,
    session_id: str | None = None,
    start_url: str | None = None,
    vision: bool = False,
    secret_scope: str = "",
) -> AgentState:
    return AgentState(
        run_id=run_id,
        task=task,
        allowed_domains=allowed_domains,
        session_id=session_id or run_id,
        start_url=start_url,
        vision=vision,
        limits=limits,
        prompt_version=prompt_version,
        secret_scope=secret_scope,
        plan=[],
        notes=None,
        observation=None,
        dom_version=None,
        current_url=None,
        elements={},
        history=[],
        findings=[],
        step=0,
        status="running",
        pending=None,
        pending_reason=None,
        human_approved=False,
        last_result=None,
        signatures=[],
        errors_in_row=0,
        stale_in_row=0,
        reflections_in_row=0,
        cost_usd=0.0,
        node_costs={},
        tokens_in=0,
        tokens_out=0,
        started_at=now,
        decided_at=now,
        result=None,
        failure_reason=None,
    )
