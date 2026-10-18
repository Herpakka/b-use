"""Runs agent runs in the background and turns graph progress into stored, published events.

Lifecycle: queued → running → (needs_human ⇄ running)* → done | failed | cancelled.
A run paused for a human holds no task; ``resume`` starts a new one from the checkpoint.

Several replicas share the store. Each active run has one owner, the replica holding a live
lease on it (``BU_RUN_LEASE_S``, renewed every third of that): it runs the run, or keeps its
paused browser session. A resume or cancel that reaches another replica is stored on the run as
a ``command`` and the owner is nudged over the bus. When an owner dies its leases run out and
any replica adopts the runs, continuing them from their checkpoints (and carrying out a waiting
command). On shutdown a replica drains, then gives its leases up so the others take over at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from datetime import datetime, timedelta
from functools import partial
from typing import Any

from langgraph.types import Command

from agent.context import AgentContext
from agent.graph import AgentGraph
from agent.nodes.finalize import own_session
from agent.runner import drive, initial_state, limits_from, new_run_id
from api import metrics
from api.schemas import CreateRunRequest, ResumeRequest, session_id
from browser.service import BrowserService, SessionNotFoundError
from core.bus import EventBus, replica_channel, run_channel
from core.config import Settings
from core.logging import bind_context, get_logger
from core.redact import redact
from core.secrets import SecretStore, secret_names
from core.store import ACTIVE, TERMINAL, RunEvent, RunRecord, RunStore, utcnow

log = get_logger("api.manager")

FINISHED = "finished"
# A run with less than this to spend can't make a single useful decision.
MIN_RUN_BUDGET_USD = 0.01
# How long a cancel forwarded to another replica is waited for before answering.
FORWARD_WAIT_S = 10.0


def month_start(now: datetime | None = None) -> datetime:
    now = now or utcnow()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


class RunError(Exception):
    def __init__(self, status: int, detail: str, *, retry_after: int | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


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
        secrets: SecretStore | None = None,
    ) -> None:
        self.store = store
        self.bus = bus
        self.browser = browser
        self.graph = graph
        self.ctx = ctx
        self.settings = settings
        self.secrets = secrets
        self.replica = settings.replica_id or socket.gethostname()
        self.draining = False
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, str] = {}  # run id → browser session id, while a task runs
        self._owned: set[str] = set()  # runs this replica holds a lease on (running or paused)
        self._cost: dict[str, float] = {}
        self._elements: dict[str, dict[str, Any]] = {}  # latest observation per run
        self._cancelling: set[str] = set()
        self._handed_over: set[str] = set()  # lease lost: another replica runs it now
        self._breakdown: dict[str, dict[str, float]] = {}
        self._loops: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------ queries
    def is_running(self, run_id: str) -> bool:
        return run_id in self._tasks

    def session_busy(self, sid: str) -> bool:
        return sid in self._sessions.values()

    @property
    def running(self) -> int:
        return len(self._tasks)

    @property
    def owned(self) -> int:
        return len(self._owned)

    @property
    def lease_s(self) -> float:
        return self.settings.run_lease_s

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Adopt runs nobody owns, then keep leases fresh and listen for forwarded commands."""
        await self.recover()
        self._loops = [
            asyncio.create_task(self._lease_loop(), name="run-leases"),
            asyncio.create_task(self._command_loop(), name="run-commands"),
        ]

    async def shutdown(self) -> None:
        """Graceful stop: refuse new runs, let running ones finish (up to ``drain_timeout_s``),
        then stop the rest — they stay ``running`` — and give every lease up, so another
        replica continues them from their checkpoints right away."""
        self.draining = True
        tasks = list(self._tasks.values())
        if tasks:
            log.info("draining", runs=len(tasks), timeout_s=self.settings.drain_timeout_s)
            _, pending = await asyncio.wait(tasks, timeout=self.settings.drain_timeout_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        for loop in self._loops:
            loop.cancel()
        await asyncio.gather(*self._loops, return_exceptions=True)
        for run_id in list(self._owned):
            with contextlib.suppress(Exception):
                await self.store.release(run_id, self.replica)
        if self._owned:
            log.info("leases_released", runs=len(self._owned))
        self._owned.clear()

    # ------------------------------------------------------------------ commands
    async def create(self, tenant: str, req: CreateRunRequest) -> RunRecord:
        if self.draining:
            raise RunError(503, "this server is shutting down; retry")
        if self.ctx is None:
            raise RunError(503, "the OpenAI key is not configured (BU_OPENAI_API_KEY)")
        if req.budget_usd is not None and req.budget_usd > self.settings.max_budget_usd:
            raise RunError(422, f"budget_usd may not exceed {self.settings.max_budget_usd}")
        await self._check_secrets(tenant, req.task)
        # Caps and limits are check-then-insert: one admission at a time, across replicas.
        try:
            async with self.store.admission():
                return await self._admit(tenant, req)
        except RunError as err:
            metrics.RUNS_REFUSED.labels(str(err.status)).inc()
            raise

    async def remaining_budget(self, tenant: str) -> float | None:
        """USD a new run of ``tenant`` may still commit this month (None: no cap applies)."""
        caps: list[float] = []
        since = month_start()
        if (cap := self.settings.tenant_monthly_budget_usd) is not None:
            caps.append(cap - (await self.store.usage(since, tenant)).committed_usd)
        if (cap := self.settings.monthly_budget_usd) is not None:
            caps.append(cap - (await self.store.usage(since)).committed_usd)
        return round(max(0.0, min(caps)), 6) if caps else None

    async def _admit(self, tenant: str, req: CreateRunRequest) -> RunRecord:
        assert self.ctx is not None
        # Every run (and every run paused for a human) holds a browser context on its replica.
        # A full replica says so at once, instead of failing the run 30 s later.
        browser = await self.browser.health()
        if browser.max_sessions and max(browser.sessions, self.owned) >= browser.max_sessions:
            raise RunError(
                503, f"all {browser.max_sessions} browsers of this replica are busy", retry_after=5
            )
        active = await self.store.count_active(tenant)
        if active >= self.settings.max_active_runs_per_tenant:
            raise RunError(429, f"{active} runs are already active for this tenant")
        budget = req.budget_usd or self.settings.budget_usd
        remaining = await self.remaining_budget(tenant)
        if remaining is not None:
            if remaining < MIN_RUN_BUDGET_USD:
                raise RunError(402, f"monthly budget exhausted (${remaining:.2f} left)")
            budget = min(budget, remaining)  # the run is cut to fit; its record shows by how much

        run_id = new_run_id()
        sid = session_id(tenant, req.session or run_id)
        if req.session and any(
            r.session_id == sid for r in await self.store.runs_with_status(ACTIVE)
        ):
            raise RunError(409, f"session {req.session!r} is in use by another active run")

        limits = limits_from(self.settings, max_steps=req.max_steps, budget_usd=budget)
        state = initial_state(
            self.ctx,
            task=req.task,
            allowed_domains=req.allowed_domains,
            run_id=run_id,
            session_id=sid,
            start_url=req.start_url,
            vision=req.vision,
            limits=limits,
            secret_scope=tenant,
        )
        run = RunRecord(
            id=run_id,
            tenant=tenant,
            task=req.task,
            allowed_domains=req.allowed_domains,
            start_url=req.start_url,
            session_id=sid,
            max_steps=limits["max_steps"],
            budget_usd=limits["budget_usd"],
            prompt_version=state["prompt_version"],
            owner=self.replica,
            lease_until=utcnow() + timedelta(seconds=self.lease_s),
        )
        await self.store.create_run(run)
        metrics.RUNS_STARTED.inc()
        self._owned.add(run_id)
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
        payload = req.model_dump()
        metrics.HUMAN_ANSWERS.labels(str(req.approved).lower()).inc()
        if await self._take(run.id):
            self._start(run, Command(resume=payload))
        else:
            # Its browser session (login window, half-filled form) lives on the owner: let the
            # owner resume it. If the owner is gone, whoever adopts the run finds the command.
            await self._forward(run, {"resume": payload})
        await self.emit(run_id, "resumed", {"approved": req.approved, "answer": req.answer[:200]})
        return run

    async def cancel(self, tenant: str, run_id: str) -> RunRecord:
        run = await self._get(tenant, run_id)
        if run.status in TERMINAL:
            raise RunError(409, f"run is already {run.status}")
        if run_id in self._owned or await self._take(run.id):
            return await self._cancel_here(run)
        await self._forward(run, {"cancel": True})
        deadline = asyncio.get_running_loop().time() + FORWARD_WAIT_S
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
            latest = await self.store.get_run(run_id)
            if latest is not None and latest.status in TERMINAL:
                return latest
        return await self._get(tenant, run_id)  # the owner (or its successor) will finish it

    async def recover(self) -> None:
        """Adopt runs whose owner is gone (no live lease) and continue them from their
        checkpoints. Runs at startup and then every lease interval, on every replica."""
        if self.ctx is None:
            return
        # Our own runs too, when this replica restarted (same id) and its leases are live.
        for orphan in await self.store.orphaned(also_owner=self.replica):
            if self.draining or orphan.id in self._owned:
                continue
            run = await self.store.claim(orphan.id, self.replica, self.lease_s)
            if run is None:
                continue  # another replica was quicker
            self._owned.add(run.id)
            if orphan.owner not in (None, self.replica):
                metrics.RUNS_ADOPTED.inc()
            await self._adopt(run)

    # ------------------------------------------------------------------ ownership internals
    async def _take(self, run_id: str) -> bool:
        if run_id in self._owned:
            return True
        if await self.store.claim(run_id, self.replica, self.lease_s) is None:
            return False
        self._owned.add(run_id)
        return True

    async def _forward(self, run: RunRecord, command: dict[str, Any]) -> None:
        await self.store.update_run(run.id, command=command)
        if run.owner:
            await self.bus.publish(replica_channel(run.owner), json.dumps({"run_id": run.id}))
        log.info("command_forwarded", run_id=run.id, owner=run.owner, op=next(iter(command)))

    async def _adopt(self, run: RunRecord) -> None:
        snapshot = await self.graph.aget_state({"configurable": {"thread_id": run.id}})
        if not snapshot.values:
            await self._finish(
                run.id, status="failed", failure_reason="lost before it started (restart)"
            )
        elif snapshot.interrupts:
            if run.status != "needs_human":
                pending = snapshot.interrupts[0].value
                await self.store.update_run(run.id, status="needs_human", pending=pending)
            await self._apply_command(run.id)
        elif not snapshot.next:
            await self._finish_from_state(run.id, dict(snapshot.values))
        elif run.command and run.command.get("cancel"):
            await self._apply_command(run.id)
        else:
            log.info("run_recovered", run_id=run.id, at=list(snapshot.next))
            await self.emit(run.id, "recovered", {"at": list(snapshot.next), "by": self.replica})
            self._start(run, None)

    async def _apply_command(self, run_id: str) -> None:
        """Carry out a resume/cancel another replica forwarded to us."""
        run = await self.store.get_run(run_id)
        if run is None or not run.command or run_id not in self._owned:
            return
        command = run.command
        await self.store.update_run(run_id, command=None)
        if command.get("cancel"):
            await self._cancel_here(run)
        elif "resume" in command and run.status == "needs_human" and not self.is_running(run_id):
            self._start(run, Command(resume=command["resume"]))

    async def _cancel_here(self, run: RunRecord) -> RunRecord:
        if task := self._tasks.get(run.id):
            self._cancelling.add(run.id)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            self._cancelling.discard(run.id)
        await self._release_session(run)
        return await self._finish(
            run.id, status="cancelled", failure_reason="cancelled by the client"
        )

    async def _lease_loop(self) -> None:
        interval = self.lease_s / 3
        while True:
            await asyncio.sleep(interval)
            try:
                await self._tend_leases()
            except Exception as err:  # ownership upkeep must never die
                log.error("lease_upkeep_failed", error=str(err)[:200])

    async def _tend_leases(self) -> None:
        owned = sorted(self._owned)
        kept = set(await self.store.renew(self.replica, owned, self.lease_s))
        for run_id in set(owned) - kept:
            self._owned.discard(run_id)
            if task := self._tasks.get(run_id):
                # Our lease ran out (we stalled) and someone else adopted the run: stop ours.
                log.warning("lease_lost", run_id=run_id)
                self._handed_over.add(run_id)
                task.cancel()
        for run_id in kept:
            await self._apply_command(run_id)  # in case the nudge was missed
        await self.recover()

    async def _command_loop(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(replica_channel(self.replica)) as sub:
                    while True:
                        message = await sub.get(1.0)
                        if message:
                            await self._apply_command(str(json.loads(message)["run_id"]))
            except asyncio.CancelledError:
                raise
            except Exception as err:
                log.warning("command_listener_failed", error=str(err)[:200])
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------------ internals
    async def _check_secrets(self, tenant: str, task: str) -> None:
        """Fail at creation, not halfway through the run, if a placeholder can't be filled."""
        names = secret_names(task)
        if not names:
            return
        if self.secrets is None:
            raise RunError(503, "the task uses secrets but BU_SECRETS_KEY is not configured")
        for name in names:
            if await self.secrets.get(tenant, name) is None:
                raise RunError(
                    422, f"unknown secret {name!r}: store it with PUT /v1/secrets/{name}"
                )

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
        """Store and publish an event. Events are a log: PII in them is masked
        (``BU_REDACT_PII``); the run record keeps the task, result and question as they are."""
        if self.settings.redact_pii:
            data = redact(data)
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
                trace_user=run.tenant,
                trace_tags=["api", self.settings.env],
            )
        except asyncio.CancelledError:
            if run.id in self._cancelling:
                return  # cancel() records the outcome
            if run.id in self._handed_over:
                self._handed_over.discard(run.id)
                return  # another replica owns it now
            raise  # shutdown: stays "running"; the lease is released and another replica adopts
        except Exception as err:
            log.exception("run_crashed")
            await self._release_session(run)
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
            metrics.HUMAN_REQUESTS.labels(str(pending.get("kind", "approval"))).inc()
            await self.emit(run.id, "needs_human", pending)
            return
        await self._finish_from_state(run.id, final)

    async def _on_event(self, run: RunRecord, node: str, update: dict[str, Any]) -> None:
        self._cost[run.id] = self._cost.get(run.id, 0.0) + float(update.get("cost_usd", 0.0))
        if costs := update.get("node_costs"):
            known = self._breakdown.setdefault(run.id, dict(run.cost_breakdown))
            for node_name, cost in costs.items():
                known[node_name] = known.get(node_name, 0.0) + float(cost)
        if "elements" in update:
            self._elements[run.id] = update["elements"]
        for type_, data in translate(node, update, self._elements.get(run.id)):
            await self.emit(run.id, type_, {**data, "cost_usd": round(self._cost[run.id], 5)})
        if node == "act" and not (update.get("last_result") or {}).get("stale"):
            await self.store.update_run(
                run.id,
                steps=int(update.get("step", 0)),
                cost_usd=self._cost[run.id],
                cost_breakdown=self._breakdown.get(run.id, run.cost_breakdown),
            )
            await self._snapshot(run)
        elif node == "decide" and update.get("status") == "needs_human":
            await self._snapshot(run)

    async def _release_session(self, run: RunRecord) -> None:
        with contextlib.suppress(Exception):
            if own_session(run.session_id, run.id):
                await self.browser.forget(run.session_id)
            else:
                await self.browser.close(run.session_id)

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
            "cost_breakdown": {k: round(v, 6) for k, v in state.get("node_costs", {}).items()},
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
        self._breakdown.pop(run_id, None)
        run = await self.store.update_run(
            run_id,
            pending=None,
            finished_at=utcnow(),
            owner=None,
            lease_until=None,
            command=None,
            **fields,
        )
        self._owned.discard(run_id)
        assert run is not None
        metrics.run_finished(run)
        return run
