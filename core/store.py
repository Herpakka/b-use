"""Run store: runs, their event log (for SSE replay) and the latest artifacts (screenshots).

``PostgresRunStore`` is the real one; ``MemoryRunStore`` has identical behaviour for tests and
single-process experiments. Agent checkpoints live separately (LangGraph's tables).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
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
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    pending: dict[str, Any] | None = None  # the interrupt payload while needs_human
    prompt_version: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


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
        "tokens_in",
        "tokens_out",
        "pending",
        "finished_at",
    }
)


class RunStore(Protocol):
    async def setup(self) -> None: ...
    async def create_run(self, run: RunRecord) -> None: ...
    async def get_run(self, run_id: str, tenant: str | None = None) -> RunRecord | None: ...
    async def list_runs(self, tenant: str, limit: int = 50) -> list[RunRecord]: ...
    async def runs_with_status(self, statuses: Sequence[str]) -> list[RunRecord]: ...
    async def count_active(self, tenant: str) -> int: ...
    async def update_run(self, run_id: str, **fields: Any) -> RunRecord | None: ...
    async def add_event(self, run_id: str, type: str, data: dict[str, Any]) -> RunEvent: ...
    async def events_after(self, run_id: str, seq: int = 0) -> list[RunEvent]: ...
    async def save_artifact(
        self, run_id: str, kind: str, content: bytes, content_type: str
    ) -> None: ...
    async def get_artifact(self, run_id: str, kind: str) -> tuple[bytes, str] | None: ...


def _check_fields(fields: dict[str, Any]) -> None:
    unknown = set(fields) - UPDATABLE
    if unknown:
        raise ValueError(f"not updatable: {sorted(unknown)}")


class MemoryRunStore:
    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.events: dict[str, list[RunEvent]] = {}
        self.artifacts: dict[tuple[str, str], tuple[bytes, str]] = {}
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
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
        self.artifacts[(run_id, kind)] = (content, content_type)

    async def get_artifact(self, run_id: str, kind: str) -> tuple[bytes, str] | None:
        return self.artifacts.get((run_id, kind))


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
CREATE INDEX IF NOT EXISTS bu_runs_tenant_created ON bu_runs (tenant, created_at DESC);
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
"""

_RUN_COLUMNS = list(RunRecord.model_fields)


def _adapt(value: Any) -> Any:
    return Jsonb(value) if isinstance(value, dict | list) else value


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

    async def count_active(self, tenant: str) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM bu_runs WHERE tenant = %s AND status = ANY(%s)",
                [tenant, list(ACTIVE)],
            )
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

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

    async def get_artifact(self, run_id: str, kind: str) -> tuple[bytes, str] | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT content, content_type FROM bu_artifacts WHERE run_id = %s AND kind = %s",
                [run_id, kind],
            )
            row = await cur.fetchone()
        return (bytes(row["content"]), str(row["content_type"])) if row else None
