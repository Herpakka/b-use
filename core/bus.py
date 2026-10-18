"""Pub/sub for live run events. Redis across processes; an in-process bus for one process/tests.

The store is the source of truth (SSE replays from it); the bus only carries the live tail, so
a dropped message costs a gap that the reader repairs from the store, never lost data.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

import redis.asyncio as aioredis

from core.logging import get_logger

log = get_logger("core.bus")


class Subscription(Protocol):
    async def get(self, wait_s: float) -> str | None:
        """Next message, or None after ``wait_s`` seconds without one."""
        ...


class EventBus(Protocol):
    async def publish(self, channel: str, message: str) -> None: ...

    def subscribe(self, channel: str) -> contextlib.AbstractAsyncContextManager[Subscription]:
        """Subscribed by the time the context is entered — replay after entering, not before."""
        ...

    async def ping(self) -> None:
        """Raise if the bus is unreachable."""
        ...

    async def close(self) -> None: ...


def replica_channel(replica: str) -> str:
    """Commands for one replica (resume/cancel of runs it owns)."""
    return f"bu:replica:{replica}"


def run_channel(run_id: str) -> str:
    return f"bu:run:{run_id}"


class _QueueSubscription:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)

    async def get(self, wait_s: float) -> str | None:
        try:
            return await asyncio.wait_for(self.queue.get(), wait_s)
        except TimeoutError:
            return None


class LocalBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[_QueueSubscription]] = {}

    async def publish(self, channel: str, message: str) -> None:
        for sub in list(self._subs.get(channel, ())):
            with contextlib.suppress(asyncio.QueueFull):  # a stuck reader repairs from the store
                sub.queue.put_nowait(message)

    @asynccontextmanager
    async def subscribe(self, channel: str) -> AsyncIterator[Subscription]:
        sub = _QueueSubscription()
        self._subs.setdefault(channel, set()).add(sub)
        try:
            yield sub
        finally:
            self._subs[channel].discard(sub)
            if not self._subs[channel]:
                del self._subs[channel]

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        self._subs.clear()


class _RedisSubscription:
    def __init__(self, pubsub: Any) -> None:
        self.pubsub = pubsub

    async def get(self, wait_s: float) -> str | None:
        message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=wait_s)
        if message is None:
            return None
        data = message["data"]
        return data.decode() if isinstance(data, bytes) else str(data)


class RedisBus:
    def __init__(self, url: str) -> None:
        self.redis: aioredis.Redis = aioredis.Redis.from_url(url)

    async def publish(self, channel: str, message: str) -> None:
        try:
            await self.redis.publish(channel, message)
        except aioredis.RedisError as err:  # live tail lost; readers recover from the store
            log.warning("bus_publish_failed", channel=channel, error=str(err)[:200])

    @asynccontextmanager
    async def subscribe(self, channel: str) -> AsyncIterator[Subscription]:
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        # SUBSCRIBE only takes effect once Redis confirms it; until then a publish from another
        # connection can slip past. Wait for the confirmation so the contract holds.
        confirmed = await pubsub.get_message(timeout=5.0)
        if confirmed is None or confirmed.get("type") != "subscribe":
            await pubsub.aclose()  # type: ignore[no-untyped-call]
            raise RuntimeError(f"Redis did not confirm the subscription to {channel}")
        try:
            yield _RedisSubscription(pubsub)
        finally:
            with contextlib.suppress(aioredis.RedisError):
                await pubsub.unsubscribe(channel)
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    async def ping(self) -> None:
        await self.redis.ping()

    async def close(self) -> None:
        await self.redis.aclose()
