"""Runs agent runs in the background and turns graph progress into stored, published events.

Lifecycle: queued → running → (needs_human ⇄ running)* → done | failed | cancelled.
A run paused for a human holds no task; ``resume`` starts a new one from the checkpoint.
On startup, runs left ``queued``/``running`` by a dead process continue from their checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
from functools import partial
from typing import Any

from langgraph.types import Command

from agent.context import AgentContext
from agent.graph import AgentGraph
from agent.runner import drive, initial_state, limits_from, new_run_id
from api.schemas import CreateRunRequest, ResumeRequest, session_id
from browser.service import BrowserService, SessionNotFoundError
from core.bus import EventBus, run_channel
from core.config import Settings
from core.logging import bind_context, get_logger
from core.store import ACTIVE, TERMINAL, RunEvent, RunRecord, RunStore, utcnow

log = get_logger("api.manager")

FINISHED = "finished"


class RunError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


MASK = "••••"


def public_args(name: str, args: dict[str, Any], elements: dict[str, Any]) -> dict[str, Any]:
    """Tool args as they may appear in events: text typed into password fields is masked."""
    if name == "TypeText":
        target = elements.get(str(args.get("index"))) or {}
        if target.get("type") == "password" and "text" in args:
            return {**args, "text": MASK}
    return args


def translate(
    node: str, update: dict[str, Any], elements: dict[str, Any] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Graph node updates → public events. Large or internal fields never leave the server.

    ``elements`` is the latest observation's element map, used to mask password input."""
    history = update.get("history") or []
    line = history[0] if history else None
    if node == "plan":
        return [("plan", {"plan": update.get("plan", [])})]
    if node == "observe":
        return [("observe", {"url": update.get("current_url")})]
    if node == "decide":
        if update.get("status") == "failed":
            return [("limit", {"reason": update.get("failure_reason")})]
        pending = update.get("pending")
        if pending:
            return [
                (
                    "decision",
                    {
                        "tool": pending["name"],
                        "args": public_args(pending["name"], pending["args"], elements or {}),
                        "needs_human": update.get("status") == "needs_human",
                        "reason": update.get("pending_reason"),
                    },
                )
            ]
        return []
    if node == "act":
        last = update.get("last_result") or {}
        if last.get("stale"):
            return [("stale", {"message": last.get("message")})]
        return [
            (
                "step",
                {
                    "step": update.get("step"),
                    "line": line,
                    "ok": last.get("ok"),
                    "url": last.get("url"),
                },
            )
        ]
    if node == "reflect":
        return [
            ("reflect", {"line": line, "advice": update.get("notes"), "plan": update.get("plan")})
        ]
    if node == "human" and line:
        return [("human", {"line": line})]
    return []


class RunManager:
    def __init__(
        self,
        *,
        store: RunStore,
        bus: EventBus,
        browser: BrowserService,
        graph: AgentGraph,
        ctx: AgentContext | None,
        settings: Settings,
    ) -> None:
        self.store = store
        self.bus = bus
        self.browser = browser
        self.graph = graph
        self.ctx = ctx
        self.settings = settings
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, str] = {}  # run id → browser session id, while a task runs
        self._cost: dict[str, float] = {}
        self._elements: dict[str, dict[str, Any]] = {}  # latest observation per run
        self._cancelling: set[str] = set()

    # ------------------------------------------------------------------ queries
    def is_running(self, run_id: str) -> bool:
        return run_id in self._tasks

    def session_busy(self, sid: str) -> bool:
        return sid in self._sessions.values()

    # ------------------------------------------------------------------ commands
    async def create(self, tenant: str, req: CreateRunRequest) -> RunRecord:
        if self.ctx is None:
            raise RunError(503, "the OpenAI key is not configured (BU_OPENAI_API_KEY)")
        if req.budget_usd is not None and req.budget_usd > self.settings.max_budget_usd:
            raise RunError(422, f"budget_usd may not exceed {self.settings.max_budget_usd}")
        active = await self.store.count_active(tenant)
        if active >= self.settings.max_active_runs_per_tenant:
            raise RunError(429, f"{active} runs are already active for this tenant")

        run_id = new_run_id()
        sid = session_id(tenant, req.session or run_id)
        if req.session and any(
            r.session_id == sid for r in await self.store.runs_with_status(ACTIVE)
        ):
            raise RunError(409, f"session {req.session!r} is in use by another active run")

        limits = limits_from(self.settings, max_steps=req.max_steps, budget_usd=req.budget_usd)
        state = initial_state(
            self.ctx,
            task=req.task,
            allowed_domains=req.allowed_domains,
            run_id=run_id,
            session_id=sid,
            start_url=req.start_url,
            vision=req.vision,
            limits=limits,
        )
        run = RunRecord(
            id=run_id,
            tenant=tenant,
            task=req.task,
            allowed_domains=req.allowed_domains,
            start_url=req.start_url,
            session_id=sid,
            max_steps=limits["max_steps"],
            prompt_version=state["prompt_version"],
        )
        await self.store.create_run(run)
        await self.emit(
            run_id,
            "created",
            {"task": req.task, "allowed_domains": req.allowed_domains, "start_url": req.start_url},
        )
        self._start(run, state)
        return run

    async def resume(self, tenant: str, run_id: str, req: ResumeRequest) -> RunRecord:
        run = await self._get(tenant, run_id)
        if run.status != "needs_human":
            raise RunError(409, f"run is {run.status}, not waiting for a human")
        if self.is_running(run_id):
            raise RunError(409, "run is already resuming")
        self._start(run, Command(resume=req.model_dump()))
        await self.emit(run_id, "resumed", {"approved": req.approved, "answer": req.answer[:200]})
        return run

    async def cancel(self, tenant: str, run_id: str) -> RunRecord:
        run = await self._get(tenant, run_id)
        if run.status in TERMINAL:
            raise RunError(409, f"run is already {run.status}")
        if task := self._tasks.get(run_id):
            self._cancelling.add(run_id)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            self._cancelling.discard(run_id)
        with contextlib.suppress(Exception):
            await self.browser.close(run.session_id)
        return await self._finish(
            run_id, status="cancelled", failure_reason="cancelled by the client"
        )

    async def recover(self) -> None:
        """Continue runs a dead process left behind, from their last checkpoint."""
        if self.ctx is None:
            return
        for run in await self.store.runs_with_status(("queued", "running")):
            snapshot = await self.graph.aget_state({"configurable": {"thread_id": run.id}})
            if not snapshot.values:
                await self._finish(
                    run.id, status="failed", failure_reason="lost before it started (restart)"
                )
            elif snapshot.interrupts:
                pending = snapshot.interrupts[0].value
                await self.store.update_run(run.id, status="needs_human", pending=pending)
            elif not snapshot.next:
                await self._finish_from_state(run.id, dict(snapshot.values))
            else:
                log.info("run_recovered", run_id=run.id, at=list(snapshot.next))
                await self.emit(run.id, "recovered", {"at": list(snapshot.next)})
                self._start(run, None)

    async def shutdown(self) -> None:
        """Stop tasks without marking runs: they are ``running`` and resume on next start."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ internals
    async def _get(self, tenant: str, run_id: str) -> RunRecord:
        run = await self.store.get_run(run_id, tenant)
        if run is None:
            raise RunError(404, "run not found")
        return run

    def _start(self, run: RunRecord, payload: Any) -> None:
        self._sessions[run.id] = run.session_id
        self._cost.setdefault(run.id, run.cost_usd)
        task = asyncio.create_task(self._drive(run, payload), name=f"run:{run.id}")
        self._tasks[run.id] = task

        def _done(_: asyncio.Task[None]) -> None:
            self._tasks.pop(run.id, None)
            self._sessions.pop(run.id, None)

        task.add_done_callback(_done)

    async def emit(self, run_id: str, type: str, data: dict[str, Any]) -> RunEvent:
        event = await self.store.add_event(run_id, type, data)
        await self.bus.publish(run_channel(run_id), event.model_dump_json())
        return event

    async def _drive(self, run: RunRecord, payload: Any) -> None:
        assert self.ctx is not None
        bind_context(run_id=run.id)
        await self.store.update_run(run.id, status="running", pending=None)
        await self.emit(run.id, "status", {"status": "running"})
        try:
            final, pending = await drive(
                self.graph,
                self.ctx,
                run.id,
                payload,
                max_steps=run.max_steps,
                on_event=partial(self._on_event, run),
            )
        except asyncio.CancelledError:
            if run.id in self._cancelling:
                return  # cancel() records the outcome
            raise  # shutdown: stays "running" and resumes on the next start
        except Exception as err:
            log.exception("run_crashed")
            with contextlib.suppress(Exception):
                await self.browser.close(run.session_id)
            await self._finish(
                run.id,
                status="failed",
                failure_reason=f"internal error: {type(err).__name__}: {str(err)[:200]}",
            )
            return
        if pending is not None:
            await self.store.update_run(
                run.id, status="needs_human", pending=pending, **self._totals(final)
            )
            await self.emit(run.id, "needs_human", pending)
            return
        await self._finish_from_state(run.id, final)

    async def _on_event(self, run: RunRecord, node: str, update: dict[str, Any]) -> None:
        self._cost[run.id] = self._cost.get(run.id, 0.0) + float(update.get("cost_usd", 0.0))
        if "elements" in update:
            self._elements[run.id] = update["elements"]
        for type_, data in translate(node, update, self._elements.get(run.id)):
            await self.emit(run.id, type_, {**data, "cost_usd": round(self._cost[run.id], 5)})
        if node == "act" and not (update.get("last_result") or {}).get("stale"):
            await self.store.update_run(
                run.id, steps=int(update.get("step", 0)), cost_usd=self._cost[run.id]
            )
            await self._snapshot(run)
        elif node == "decide" and update.get("status") == "needs_human":
            await self._snapshot(run)

    async def _snapshot(self, run: RunRecord) -> None:
        """Keep the latest screenshot so it can be served after the session is gone."""
        try:
            image = await self.browser.screenshot(run.session_id)
        except SessionNotFoundError:
            return
        except Exception as err:
            log.warning("screenshot_failed", error=str(err)[:200])
            return
        await self.store.save_artifact(run.id, "screenshot", image, "image/jpeg")

    @staticmethod
    def _totals(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "steps": int(state.get("step", 0)),
            "cost_usd": float(state.get("cost_usd", 0.0)),
            "tokens_in": int(state.get("tokens_in", 0)),
            "tokens_out": int(state.get("tokens_out", 0)),
        }

    async def _finish_from_state(self, run_id: str, state: dict[str, Any]) -> RunRecord:
        return await self._finish(
            run_id,
            status="done" if state.get("status") == "done" else "failed",
            result=state.get("result"),
            failure_reason=state.get("failure_reason"),
            **self._totals(state),
        )

    async def _finish(self, run_id: str, **fields: Any) -> RunRecord:
        # The event goes first: a reader that sees a terminal status can rely on the
        # "finished" event already being in the log.
        await self.emit(
            run_id,
            FINISHED,
            {k: fields.get(k) for k in ("status", "result", "failure_reason", "steps", "cost_usd")},
        )
        self._cost.pop(run_id, None)
        self._elements.pop(run_id, None)
        run = await self.store.update_run(run_id, pending=None, finished_at=utcnow(), **fields)
        assert run is not None
        return run
