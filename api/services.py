"""Composition root: builds the real services for the app's lifetime."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from agent.context import AgentContext
from agent.graph import compile_graph
from agent.llm import make_chat_model
from api.janitor import Janitor
from api.manager import RunManager
from browser.local import local_browser
from browser.service import BrowserService
from core.bus import EventBus, LocalBus, RedisBus
from core.config import Settings
from core.costs import PriceTable
from core.logging import get_logger
from core.ratelimit import LocalRateLimiter, RateLimiter, RedisRateLimiter
from core.secrets import MemorySecretStore, PostgresSecretStore, SecretCipher, SecretStore
from core.store import PostgresRunStore, RunStore

log = get_logger("api.services")


@dataclass
class Services:
    settings: Settings
    store: RunStore
    bus: EventBus
    limiter: RateLimiter
    browser: BrowserService
    manager: RunManager
    secrets: SecretStore = field(default_factory=MemorySecretStore)


ServicesFactory = Callable[[Settings], AbstractAsyncContextManager[Services]]


def agent_context(
    settings: Settings, browser: BrowserService, secrets: SecretStore | None = None
) -> AgentContext | None:
    if settings.openai_api_key is None:
        log.warning("openai_key_missing", hint="runs are refused until BU_OPENAI_API_KEY is set")
        return None
    prices = PriceTable.from_settings(settings)
    prices.require(settings.model_decide, settings.model_small)
    return AgentContext(
        browser=browser,
        llm_decide=make_chat_model(settings, settings.model_decide),
        llm_small=make_chat_model(settings, settings.model_small),
        settings=settings,
        prices=prices,
        secrets=secrets.get if secrets is not None else None,
    )


@asynccontextmanager
async def build_services(settings: Settings) -> AsyncIterator[Services]:
    async with AsyncExitStack() as stack:
        # One pool for the run store and LangGraph's checkpointer (which needs exactly these
        # connection settings).
        pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open(wait=True, timeout=30)
        stack.push_async_callback(pool.close)
        store = PostgresRunStore(pool)
        await store.setup()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        cipher = SecretCipher.from_setting(settings.secrets_key)
        secrets = PostgresSecretStore(pool, cipher)
        await secrets.setup()
        if cipher is None:
            log.info("secrets_disabled", hint="set BU_SECRETS_KEY to use {{SECRET:name}}")

        bus: EventBus
        limiter: RateLimiter
        if settings.event_bus == "redis":
            redis_bus = RedisBus(settings.redis_url)
            bus, limiter = redis_bus, RedisRateLimiter(redis_bus.redis)
        else:
            bus, limiter = LocalBus(), LocalRateLimiter()
        stack.push_async_callback(bus.close)

        browser = await stack.enter_async_context(local_browser(settings))
        manager = RunManager(
            store=store,
            bus=bus,
            browser=browser,
            graph=compile_graph(saver),
            ctx=agent_context(settings, browser, secrets if secrets.available else None),
            settings=settings,
            secrets=secrets if secrets.available else None,
        )
        stack.push_async_callback(manager.shutdown)
        await manager.recover()
        janitor = Janitor(store, saver, settings)
        janitor.start()
        stack.push_async_callback(janitor.stop)
        yield Services(
            settings=settings,
            store=store,
            bus=bus,
            limiter=limiter,
            browser=browser,
            manager=manager,
            secrets=secrets,
        )
