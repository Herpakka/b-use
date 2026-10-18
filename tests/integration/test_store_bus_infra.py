"""PostgresRunStore against real Postgres, RedisBus/RedisRateLimiter against real Redis."""

from __future__ import annotations

import os
import secrets
import sys
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

from core.bus import RedisBus
from core.ratelimit import RedisRateLimiter
from core.secrets import (
    PostgresSecretStore,
    Secret,
    SecretCipher,
    SecretsUnavailableError,
    generate_key,
)
from core.store import PostgresRunStore, RunRecord, utcnow


def _skip_on_windows() -> None:
    if sys.platform == "win32":
        pytest.skip("async psycopg cannot use the Windows Proactor loop; run in the container")


@pytest.fixture
async def pg_store() -> AsyncIterator[PostgresRunStore]:
    _skip_on_windows()
    url = os.environ.get("BU_DATABASE_URL")
    if not url:
        pytest.skip("BU_DATABASE_URL not set")
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        url,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    try:
        await pool.open(wait=True, timeout=3)
    except Exception as err:
        pytest.skip(f"Postgres unreachable: {err}")
    store = PostgresRunStore(pool)
    await store.setup()
    yield store
    await pool.close()


@pytest.fixture
async def redis_bus() -> AsyncIterator[RedisBus]:
    url = os.environ.get("BU_REDIS_URL")
    if not url:
        pytest.skip("BU_REDIS_URL not set")
    bus = RedisBus(url)
    try:
        await bus.redis.ping()
    except Exception as err:
        await bus.close()
        pytest.skip(f"Redis unreachable: {err}")
    yield bus
    await bus.close()


def _run(tenant: str) -> RunRecord:
    rid = f"run-{secrets.token_hex(6)}"
    return RunRecord(
        id=rid,
        tenant=tenant,
        task="t",
        allowed_domains=["a.test"],
        session_id=f"{tenant}.{rid}",
        max_steps=10,
    )


@pytest.mark.pg
async def test_postgres_store_round_trip(pg_store: PostgresRunStore) -> None:
    tenant = f"t{secrets.token_hex(4)}"
    run = _run(tenant)
    await pg_store.create_run(run)
    assert (await pg_store.get_run(run.id, tenant)) is not None
    assert await pg_store.get_run(run.id, "someone-else") is None
    assert await pg_store.count_active(tenant) == 1

    updated = await pg_store.update_run(
        run.id, status="needs_human", pending={"kind": "approval", "action": "Click [2]"}, steps=3
    )
    assert updated is not None and updated.pending == {"kind": "approval", "action": "Click [2]"}
    assert updated.updated_at > run.updated_at
    with pytest.raises(ValueError):
        await pg_store.update_run(run.id, tenant="evil")

    events = [await pg_store.add_event(run.id, f"e{i}", {"i": i, "ไทย": "ได้"}) for i in range(3)]
    assert [e.seq for e in events] == [1, 2, 3]
    assert [e.data["i"] for e in await pg_store.events_after(run.id, 1)] == [1, 2]

    await pg_store.save_artifact(run.id, "screenshot", b"one", "image/jpeg")
    await pg_store.save_artifact(run.id, "screenshot", b"two", "image/jpeg")
    assert await pg_store.get_artifact(run.id, "screenshot") == (b"two", "image/jpeg")

    await pg_store.update_run(run.id, status="done", finished_at=utcnow())
    assert await pg_store.count_active(tenant) == 0
    assert [r.id for r in await pg_store.list_runs(tenant)] == [run.id]
    assert run.id in {r.id for r in await pg_store.runs_with_status(["done"])}


@pytest.mark.redis
async def test_redis_bus_delivers_after_subscribe(redis_bus: RedisBus) -> None:
    channel = f"bu:test:{secrets.token_hex(4)}"
    async with redis_bus.subscribe(channel) as sub:
        await redis_bus.publish(channel, "hello")
        assert await sub.get(2.0) == "hello"
        assert await sub.get(0.2) is None  # timeout → None, used for SSE heartbeats


@pytest.mark.redis
async def test_redis_rate_limiter(redis_bus: RedisBus) -> None:
    limiter = RedisRateLimiter(redis_bus.redis)
    key = f"test-{secrets.token_hex(4)}"
    results = [await limiter.hit(key, 2, 60) for _ in range(3)]
    assert [ok for ok, _ in results] == [True, True, False]
    assert 1 <= results[-1][1] <= 60


@pytest.mark.pg
async def test_postgres_secret_store_encrypts_at_rest(pg_store: PostgresRunStore) -> None:
    cipher = SecretCipher.from_setting(SecretStr(generate_key()))
    store = PostgresSecretStore(pg_store.pool, cipher)
    await store.setup()
    scope = f"t-{secrets.token_hex(4)}"
    await store.put(scope, "pw", "correct-horse-battery", ["Shop.Test"])
    await store.put(scope, "pw", "correct-horse-battery-2", ["shop.test"])  # replaces
    assert await store.get(scope, "pw") == Secret("pw", "correct-horse-battery-2", ("shop.test",))
    assert await store.get("other-" + scope, "pw") is None
    assert [s.name for s in await store.list(scope)] == ["pw"]

    async with pg_store.pool.connection() as conn:
        cur = await conn.execute("SELECT ciphertext FROM bu_secrets WHERE scope = %s", [scope])
        row = await cur.fetchone()
    assert row is not None and b"correct-horse" not in bytes(row["ciphertext"])

    # A different key (e.g. rotated without re-encrypting) reads nothing rather than garbage.
    other = PostgresSecretStore(pg_store.pool, SecretCipher.from_setting(SecretStr(generate_key())))
    assert await other.get(scope, "pw") is None
    with pytest.raises(SecretsUnavailableError):
        await PostgresSecretStore(pg_store.pool, None).put(scope, "x", "long enough", [])
    assert await store.delete(scope, "pw")


@pytest.mark.pg
async def test_postgres_usage_counts_active_runs_at_budget(pg_store: PostgresRunStore) -> None:
    tenant = f"u{secrets.token_hex(4)}"
    since = utcnow() - timedelta(minutes=1)
    for i, (status, cost, budget) in enumerate([("done", 0.1, 0.15), ("running", 0.02, 0.15)]):
        await pg_store.create_run(
            RunRecord(
                id=f"run-{tenant}{i}",
                tenant=tenant,
                task="t",
                allowed_domains=["a.test"],
                session_id=f"{tenant}.s",
                max_steps=5,
                status=status,
                cost_usd=cost,
                budget_usd=budget,
                tokens_in=100,
                cost_breakdown={"decide": cost},
            )
        )
    usage = await pg_store.usage(since, tenant)
    assert usage.runs == 2 and usage.tokens_in == 200
    assert usage.cost_usd == pytest.approx(0.12)
    assert usage.committed_usd == pytest.approx(0.25)
    assert (await pg_store.usage(utcnow() + timedelta(minutes=1), tenant)).runs == 0
    run = await pg_store.get_run(f"run-{tenant}0")
    assert run is not None and run.cost_breakdown == {"decide": 0.1}


@pytest.mark.pg
async def test_postgres_retention_queries(pg_store: PostgresRunStore) -> None:
    tag = secrets.token_hex(4)
    now = utcnow()
    for i, (status, ended) in enumerate([("done", 2.0), ("running", None), ("failed", 40.0)]):
        run_id = f"run-{tag}{i}"
        await pg_store.create_run(
            RunRecord(
                id=run_id,
                tenant=f"r{tag}",
                task="t",
                allowed_domains=["a.test"],
                session_id=f"r{tag}.s",
                max_steps=5,
                status=status,
                finished_at=None if ended is None else now - timedelta(days=ended),
            )
        )
        await pg_store.add_event(run_id, "created", {})
        await pg_store.save_artifact(run_id, "screenshot", b"x", "image/jpeg")

    mine = {f"run-{tag}0", f"run-{tag}2"}
    to_purge = set(await pg_store.runs_to_purge(now - timedelta(days=1), 10_000))
    assert to_purge >= mine and f"run-{tag}1" not in to_purge
    await pg_store.update_run(f"run-{tag}0", purged_at=now)
    assert f"run-{tag}0" not in await pg_store.runs_to_purge(now - timedelta(days=1), 10_000)

    await pg_store.delete_artifacts_before(now - timedelta(hours=1))
    assert await pg_store.get_artifact(f"run-{tag}1", "screenshot") is not None  # fresh
    assert await pg_store.delete_runs_before(now - timedelta(days=30)) >= 1
    assert await pg_store.get_run(f"run-{tag}2") is None
    assert await pg_store.events_after(f"run-{tag}2") == []  # cascaded
    assert await pg_store.get_run(f"run-{tag}1") is not None
