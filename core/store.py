"""Run store: runs, their event log (for SSE replay) and the latest artifacts (screenshots).

``PostgresRunStore`` is the real one; ``MemoryRunStore`` has identical behaviour for tests and
single-process experiments. Agent checkpoints live separately (LangGraph's tables).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

RunStatus = Literal["queued", "running", "needs_human", "done", "failed", "cancelled"]
TERMINAL: frozenset[str] = frozenset({"done", "failed", "cancelled"})
ACTIVE: tuple[str, ...] = ("queued", "running", "needs_human")


def utcnow() -> datetime:
    return datetime.now(UTC)


class RunRecord(BaseModel):
    id: str
    tenant: str
    task: str
    allowed_domains: list[str]
    start_url: str | None = None
    session_id: str
    max_steps: int
    status: RunStatus = "queued"
    result: str | None = None
    failure_reason: str | None = None
    steps: int = 0
    budget_usd: float = 0.0  # the run's own cap, reserved against monthly budgets
    cost_usd: float = 0.0
    cost_breakdown: dict[str, float] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    pending: dict[str, Any] | None = None  # the interrupt payload while needs_human
    prompt_version: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    purged_at: datetime | None = None  # when its agent checkpoints were deleted
    # Multi-replica ownership: the replica holding a live lease runs the run (or keeps its
    # paused browser session); ``command`` is a resume/cancel waiting for that replica.
    owner: str | None = None
    lease_until: datetime | None = None
    command: dict[str, Any] | None = None


class RunEvent(BaseModel):
    run_id: str
    seq: int
    type: str
    data: dict[str, Any]
    created_at: datetime = Field(default_factory=utcnow)


UPDATABLE = frozenset(
    {
        "status",
        "result",
        "failure_reason",
        "steps",
        "cost_usd",
        "cost_breakdown",
        "tokens_in",
        "tokens_out",
        "pending",
        "finished_at",
        "purged_at",
        "owner",
        "lease_until",
        "command",
    }
)


class Usage(BaseModel):
    runs: int = 0
    cost_usd: float = 0.0
    # Spent by finished runs + the full budget of active ones: what a new run must fit beside.
    committed_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


def committed(run: RunRecord) -> float:
    return max(run.budget_usd, run.cost_usd) if run.status in ACTIVE else run.cost_usd


class RunStore(Protocol):
    async def setup(self) -> None: ...
    async def ping(self) -> None: ...
    async def create_run(self, run: RunRecord) -> None: ...
    async def get_run(self, run_id: str, tenant: str | None = None) -> RunRecord | None: ...
    async def list_runs(self, tenant: str, limit: int = 50) -> list[RunRecord]: ...
    async def runs_with_status(self, statuses: Sequence[str]) -> list[RunRecord]: ...
    async def count_active(self, tenant: str) -> int: ...
    # ownership (api.manager)
    async def claim(self, run_id: str, owner: str, lease_s: float) -> RunRecord | None: ...
    async def renew(self, owner: str, run_ids: Sequence[str], lease_s: float) -> list[str]: ...
    async def release(self, run_id: str, owner: str) -> None: ...
    async def orphaned(self, also_owner: str | None = None) -> list[RunRecord]: ...
    def admission(self) -> AbstractAsyncContextManager[None]: ...
    async def usage(self, since: datetime, tenant: str | None = None) -> Usage: ...
    async def update_run(self, run_id: str, **fields: Any) -> RunRecord | None: ...
    async def add_event(self, run_id: str, type: str, data: dict[str, Any]) -> RunEvent: ...
    async def events_after(self, run_id: str, seq: int = 0) -> list[RunEvent]: ...
    async def save_artifact(
        self, run_id: str, kind: str, content: bytes, content_type: str
    ) -> None: ...
    async def get_artifact(self, run_id: str, kind: str) -> tuple[bytes, str] | None: ...
    # retention (api.janitor)
    async def delete_artifacts_before(self, cutoff: datetime) -> int: ...
    async def runs_to_purge(self, finished_before: datetime, limit: int) -> list[str]: ...
    async def delete_runs_before(self, finished_before: datetime) -> int: ...


def _check_fields(fields: dict[str, Any]) -> None:
    unknown = set(fields) - UPDATABLE
    if unknown:
        raise ValueError(f"not updatable: {sorted(unknown)}")


class MemoryRunStore:
    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.events: dict[str, list[RunEvent]] = {}
        self.artifacts: dict[tuple[str, str], tuple[bytes, str, datetime]] = {}
        self._lock = asyncio.Lock()
        self._admission = asyncio.Lock()

    async def setup(self) -> None:
        return None

    async def ping(self) -> None:
        return None

    async def create_run(self, run: RunRecord) -> None:
        self.runs[run.id] = run.model_copy()
        self.events[run.id] = []

    async def get_run(self, run_id: str, tenant: str | None = None) -> RunRecord | None:
        run = self.runs.get(run_id)
        if run is None or (tenant is not None and run.tenant != tenant):
            return None
        return run.model_copy()

    async def list_runs(self, tenant: str, limit: int = 50) -> list[RunRecord]:
        mine = [r for r in self.runs.values() if r.tenant == tenant]
        return [r.model_copy() for r in sorted(mine, key=lambda r: r.created_at, reverse=True)][
            :limit
        ]

    async def runs_with_status(self, statuses: Sequence[str]) -> list[RunRecord]:
        return [r.model_copy() for r in self.runs.values() if r.status in statuses]

    async def count_active(self, tenant: str) -> int:
        return sum(1 for r in self.runs.values() if r.tenant == tenant and r.status in ACTIVE)

    def _leasable(self, run: RunRecord, owner: str) -> bool:
        live = run.lease_until is not None and run.lease_until >= utcnow()
        return run.status in ACTIVE and (run.owner in (None, owner) or not live)

    async def claim(self, run_id: str, owner: str, lease_s: float) -> RunRecord | None:
        async with self._lock:
            run = self.runs.get(run_id)
            if run is None or not self._leasable(run, owner):
                return None
            until = utcnow() + timedelta(seconds=lease_s)
            self.runs[run_id] = run.model_copy(update={"owner": owner, "lease_until": until})
            return self.runs[run_id].model_copy()

    async def renew(self, owner: str, run_ids: Sequence[str], lease_s: float) -> list[str]:
        kept = []
        until = utcnow() + timedelta(seconds=lease_s)
        for run_id in run_ids:
            run = self.runs.get(run_id)
            if run is not None and run.owner == owner and run.status in ACTIVE:
                self.runs[run_id] = run.model_copy(update={"lease_until": until})
                kept.append(run_id)
        return kept

    async def release(self, run_id: str, owner: str) -> None:
        run = self.runs.get(run_id)
        if run is not None and run.owner == owner:
            self.runs[run_id] = run.model_copy(update={"owner": None, "lease_until": None})

    async def orphaned(self, also_owner: str | None = None) -> list[RunRecord]:
        now = utcnow()
        return [
            r.model_copy()
            for r in self.runs.values()
            if r.status in ACTIVE
            and (
                r.owner is None
                or r.lease_until is None
                or r.lease_until < now
                or (also_owner is not None and r.owner == also_owner)
            )
        ]

    @asynccontextmanager
    async def admission(self) -> AsyncIterator[None]:
        async with self._admission:
            yield

    async def usage(self, since: datetime, tenant: str | None = None) -> Usage:
        runs = [
            r
            for r in self.runs.values()
            if r.created_at >= since and (tenant is None or r.tenant == tenant)
        ]
        return Usage(
            runs=len(runs),
            cost_usd=sum(r.cost_usd for r in runs),
            committed_usd=sum(committed(r) for r in runs),
            tokens_in=sum(r.tokens_in for r in runs),
            tokens_out=sum(r.tokens_out for r in runs),
        )

    async def update_run(self, run_id: str, **fields: Any) -> RunRecord | None:
        _check_fields(fields)
        run = self.runs.get(run_id)
        if run is None:
            return None
        updated = run.model_copy(update={**fields, "updated_at": utcnow()})
        self.runs[run_id] = updated
        return updated.model_copy()

    async def add_event(self, run_id: str, type: str, data: dict[str, Any]) -> RunEvent:
        async with self._lock:
            log = self.events.setdefault(run_id, [])
            event = RunEvent(run_id=run_id, seq=len(log) + 1, type=type, data=data)
            log.append(event)
            return event

    async def events_after(self, run_id: str, seq: int = 0) -> list[RunEvent]:
        return [e for e in self.events.get(run_id, []) if e.seq > seq]

    async def save_artifact(
        self, run_id: str, kind: str, content: bytes, content_type: str
    ) -> None:
        self.artifacts[(run_id, kind)] = (content, content_type, utcnow())

    async def get_artifact(self, run_id: str, kind: str) -> tuple[bytes, str] | None:
        found = self.artifacts.get((run_id, kind))
        return (found[0], found[1]) if found else None

    async def delete_artifacts_before(self, cutoff: datetime) -> int:
        old = [key for key, (_, _, at) in self.artifacts.items() if at < cutoff]
        for key in old:
            del self.artifacts[key]
        return len(old)

    async def runs_to_purge(self, finished_before: datetime, limit: int) -> list[str]:
        return [
            r.id
            for r in self.runs.values()
            if r.status in TERMINAL
            and r.finished_at is not None
            and r.finished_at < finished_before
            and r.purged_at is None
        ][:limit]

    async def delete_runs_before(self, finished_before: datetime) -> int:
        old = [
            r.id
            for r in self.runs.values()
            if r.status in TERMINAL
            and r.finished_at is not None
            and r.finished_at < finished_before
        ]
        for run_id in old:
            del self.runs[run_id]
            self.events.pop(run_id, None)
            for key in [k for k in self.artifacts if k[0] == run_id]:
                del self.artifacts[key]
        return len(old)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS bu_runs (
    id              text PRIMARY KEY,
    tenant          text NOT NULL,
    task            text NOT NULL,
    allowed_domains jsonb NOT NULL,
    start_url       text,
    session_id      text NOT NULL,
    max_steps       integer NOT NULL,
    status          text NOT NULL,
    result          text,
    failure_reason  text,
    steps           integer NOT NULL DEFAULT 0,
    cost_usd        double precision NOT NULL DEFAULT 0,
    tokens_in       integer NOT NULL DEFAULT 0,
    tokens_out      integer NOT NULL DEFAULT 0,
    pending         jsonb,
    prompt_version  text NOT NULL DEFAULT '',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz
);
ALTER TABLE bu_runs ADD COLUMN IF NOT EXISTS budget_usd double precision NOT NULL DEFAULT 0;
ALTER TABLE bu_runs ADD COLUMN IF NOT EXISTS cost_breakdown jsonb NOT NULL DEFAULT '{}';
ALTER TABLE bu_runs ADD COLUMN IF NOT EXISTS purged_at timestamptz;
ALTER TABLE bu_runs ADD COLUMN IF NOT EXISTS owner text;
ALTER TABLE bu_runs ADD COLUMN IF NOT EXISTS lease_until timestamptz;
ALTER TABLE bu_runs ADD COLUMN IF NOT EXISTS command jsonb;
CREATE INDEX IF NOT EXISTS bu_runs_tenant_created ON bu_runs (tenant, created_at DESC);
CREATE INDEX IF NOT EXISTS bu_runs_created ON bu_runs (created_at);
CREATE INDEX IF NOT EXISTS bu_runs_finished ON bu_runs (finished_at);
CREATE INDEX IF NOT EXISTS bu_runs_status ON bu_runs (status);
CREATE TABLE IF NOT EXISTS bu_run_events (
    run_id     text NOT NULL REFERENCES bu_runs (id) ON DELETE CASCADE,
    seq        integer NOT NULL,
    type       text NOT NULL,
    data       jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS bu_artifacts (
    run_id       text NOT NULL REFERENCES bu_runs (id) ON DELETE CASCADE,
    kind         text NOT NULL,
    content      bytea NOT NULL,
    content_type text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, kind)
);
CREATE INDEX IF NOT EXISTS bu_artifacts_created ON bu_artifacts (created_at);
"""

_RUN_COLUMNS = list(RunRecord.model_fields)
ADMISSION_LOCK = 0x62755F61646D  # "bu_adm"


def _adapt(value: Any) -> Any:
    return Jsonb(value) if isinstance(value, dict | list) else value


@asynccontextmanager
async def advisory_lock(conn: Any, key: int, poll_s: float = 0.2) -> AsyncIterator[None]:
    """Hold a Postgres session-level advisory lock on ``conn``.

    Waits by polling ``pg_try_advisory_lock`` rather than blocking in ``pg_advisory_lock``: a
    blocked call is an open statement, and ``CREATE INDEX CONCURRENTLY`` (LangGraph's schema
    setup) waits for every open statement — while the lock holder waits for that index. Two
    replicas starting together deadlocked exactly like that.
    """
    while True:
        cur = await conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", [key])
        row = await cur.fetchone()
        if row and row["ok"]:
            break
        await asyncio.sleep(poll_s)
    try:
        yield
    finally:
        await conn.execute("SELECT pg_advisory_unlock(%s)", [key])


class PostgresRunStore:
    """Expects a pool whose connections use ``autocommit=True`` and ``row_factory=dict_row``
    (the same pool LangGraph's AsyncPostgresSaver needs)."""

    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self.pool = pool
        # Event sequence numbers are assigned per run; a lock keeps two writers in this
        # process from racing (the primary key catches anything else).
        self._event_lock = asyncio.Lock()

    async def setup(self) -> None:
        # One statement per execute: with prepare_threshold=0 (needed by LangGraph's saver on
        # the shared pool) every query is prepared, and prepared statements can't hold several.
        async with self.pool.connection() as conn:
            for statement in filter(None, (part.strip() for part in _SCHEMA.split(";"))):
                await conn.execute(statement)

    async def ping(self) -> None:
        async with self.pool.connection(timeout=3) as conn:
            await conn.execute("SELECT 1")

    async def create_run(self, run: RunRecord) -> None:
        data = run.model_dump()
        query = sql.SQL("INSERT INTO bu_runs ({}) VALUES ({})").format(
            sql.SQL(", ").join(map(sql.Identifier, _RUN_COLUMNS)),
            sql.SQL(", ").join(sql.Placeholder() * len(_RUN_COLUMNS)),
        )
        async with self.pool.connection() as conn:
            await conn.execute(query, [_adapt(data[c]) for c in _RUN_COLUMNS])

    async def _fetch_runs(self, where: sql.Composable, params: Sequence[Any]) -> list[RunRecord]:
        query = sql.SQL("SELECT * FROM bu_runs WHERE {} ORDER BY created_at DESC").format(where)
        async with self.pool.connection() as conn:
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()
        return [RunRecord.model_validate(row) for row in rows]

    async def get_run(self, run_id: str, tenant: str | None = None) -> RunRecord | None:
        if tenant is None:
            runs = await self._fetch_runs(sql.SQL("id = %s"), [run_id])
        else:
            runs = await self._fetch_runs(sql.SQL("id = %s AND tenant = %s"), [run_id, tenant])
        return runs[0] if runs else None

    async def list_runs(self, tenant: str, limit: int = 50) -> list[RunRecord]:
        return (await self._fetch_runs(sql.SQL("tenant = %s"), [tenant]))[:limit]

    async def runs_with_status(self, statuses: Sequence[str]) -> list[RunRecord]:
        return await self._fetch_runs(sql.SQL("status = ANY(%s)"), [list(statuses)])

    async def claim(self, run_id: str, owner: str, lease_s: float) -> RunRecord | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE bu_runs SET owner = %s, lease_until = now() + make_interval(secs => %s)
                WHERE id = %s AND status = ANY(%s)
                  AND (owner IS NULL OR owner = %s OR lease_until IS NULL OR lease_until < now())
                RETURNING *
                """,
                [owner, lease_s, run_id, list(ACTIVE), owner],
            )
            row = await cur.fetchone()
        return RunRecord.model_validate(row) if row else None

    async def renew(self, owner: str, run_ids: Sequence[str], lease_s: float) -> list[str]:
        if not run_ids:
            return []
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "UPDATE bu_runs SET lease_until = now() + make_interval(secs => %s)"
                " WHERE owner = %s AND id = ANY(%s) AND status = ANY(%s) RETURNING id",
                [lease_s, owner, list(run_ids), list(ACTIVE)],
            )
            rows = await cur.fetchall()
        return [str(r["id"]) for r in rows]

    async def release(self, run_id: str, owner: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE bu_runs SET owner = NULL, lease_until = NULL WHERE id = %s AND owner = %s",
                [run_id, owner],
            )

    async def orphaned(self, also_owner: str | None = None) -> list[RunRecord]:
        """Active runs with no live lease (plus those of ``also_owner``)."""
        return await self._fetch_runs(
            sql.SQL(
                "status = ANY(%s) AND (owner IS NULL OR lease_until IS NULL"
                " OR lease_until < now() OR owner = %s)"
            ),
            [list(ACTIVE), also_owner or ""],
        )

    @asynccontextmanager
    async def admission(self) -> AsyncIterator[None]:
        """Serialise run admission across replicas (caps and limits are check-then-insert)."""
        async with self.pool.connection() as conn, advisory_lock(conn, ADMISSION_LOCK):
            yield

    async def count_active(self, tenant: str) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM bu_runs WHERE tenant = %s AND status = ANY(%s)",
                [tenant, list(ACTIVE)],
            )
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def usage(self, since: datetime, tenant: str | None = None) -> Usage:
        where: sql.Composable = sql.SQL("created_at >= %s")
        params: list[Any] = [since]
        if tenant is not None:
            where = sql.SQL("{} AND tenant = %s").format(where)
            params.append(tenant)
        query = sql.SQL(
            """
            SELECT count(*) AS runs,
                   COALESCE(sum(cost_usd), 0) AS cost_usd,
                   COALESCE(sum(CASE WHEN status = ANY(%s) THEN GREATEST(budget_usd, cost_usd)
                                     ELSE cost_usd END), 0) AS committed_usd,
                   COALESCE(sum(tokens_in), 0) AS tokens_in,
                   COALESCE(sum(tokens_out), 0) AS tokens_out
            FROM bu_runs WHERE {}
            """
        ).format(where)
        async with self.pool.connection() as conn:
            cur = await conn.execute(query, [list(ACTIVE), *params])
            row = await cur.fetchone()
        return Usage.model_validate(row or {})

    async def update_run(self, run_id: str, **fields: Any) -> RunRecord | None:
        _check_fields(fields)
        if not fields:
            return await self.get_run(run_id)
        assignments = sql.SQL(", ").join(
            sql.SQL("{} = %s").format(sql.Identifier(k)) for k in fields
        )
        query = sql.SQL(
            "UPDATE bu_runs SET {}, updated_at = now() WHERE id = %s RETURNING *"
        ).format(assignments)
        async with self.pool.connection() as conn:
            cur = await conn.execute(query, [*(_adapt(v) for v in fields.values()), run_id])
            row = await cur.fetchone()
        return RunRecord.model_validate(row) if row else None

    async def add_event(self, run_id: str, type: str, data: dict[str, Any]) -> RunEvent:
        async with self._event_lock, self.pool.connection() as conn:
            cur = await conn.execute(
                """
                INSERT INTO bu_run_events (run_id, seq, type, data)
                VALUES (%s, (SELECT COALESCE(MAX(seq), 0) + 1 FROM bu_run_events
                             WHERE run_id = %s), %s, %s)
                RETURNING run_id, seq, type, data, created_at
                """,
                [run_id, run_id, type, Jsonb(data)],
            )
            row = await cur.fetchone()
        assert row is not None
        return RunEvent.model_validate(row)

    async def events_after(self, run_id: str, seq: int = 0) -> list[RunEvent]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT run_id, seq, type, data, created_at FROM bu_run_events"
                " WHERE run_id = %s AND seq > %s ORDER BY seq",
                [run_id, seq],
            )
            rows = await cur.fetchall()
        return [RunEvent.model_validate(row) for row in rows]

    async def save_artifact(
        self, run_id: str, kind: str, content: bytes, content_type: str
    ) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO bu_artifacts (run_id, kind, content, content_type)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id, kind) DO UPDATE
                SET content = EXCLUDED.content, content_type = EXCLUDED.content_type,
                    created_at = now()
                """,
                [run_id, kind, content, content_type],
            )

    async def delete_artifacts_before(self, cutoff: datetime) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute("DELETE FROM bu_artifacts WHERE created_at < %s", [cutoff])
            return int(cur.rowcount)

    async def runs_to_purge(self, finished_before: datetime, limit: int) -> list[str]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT id FROM bu_runs WHERE status = ANY(%s) AND finished_at < %s"
                " AND purged_at IS NULL ORDER BY finished_at LIMIT %s",
                [list(TERMINAL), finished_before, limit],
            )
            rows = await cur.fetchall()
        return [str(row["id"]) for row in rows]

    async def delete_runs_before(self, finished_before: datetime) -> int:
        # Events and artifacts go with their run (ON DELETE CASCADE).
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM bu_runs WHERE status = ANY(%s) AND finished_at < %s",
                [list(TERMINAL), finished_before],
            )
            return int(cur.rowcount)

    async def get_artifact(self, run_id: str, kind: str) -> tuple[bytes, str] | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT content, content_type FROM bu_artifacts WHERE run_id = %s AND kind = %s",
                [run_id, kind],
            )
            row = await cur.fetchone()
        return (bytes(row["content"]), str(row["content_type"])) if row else None
