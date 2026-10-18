"""Fixed-window rate limiting per API key/tenant."""

from __future__ import annotations

import time
from typing import Protocol

import redis.asyncio as aioredis

from core.logging import get_logger

log = get_logger("core.ratelimit")


class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int, window_s: int) -> tuple[bool, int]:
        """Count one request. Returns (allowed, seconds until the window resets)."""
        ...


def _window(window_s: int, now: float) -> tuple[int, int]:
    start = int(now // window_s) * window_s
    return start, max(1, start + window_s - int(now))


class LocalRateLimiter:
    def __init__(self) -> None:
        self._counts: dict[tuple[str, int], int] = {}

    async def hit(self, key: str, limit: int, window_s: int) -> tuple[bool, int]:
        start, retry = _window(window_s, time.time())
        self._counts = {k: v for k, v in self._counts.items() if k[1] >= start}  # drop old windows
        n = self._counts.get((key, start), 0) + 1
        self._counts[(key, start)] = n
        return n <= limit, retry


class RedisRateLimiter:
    def __init__(self, redis: aioredis.Redis) -> None:
        self.redis = redis

    async def hit(self, key: str, limit: int, window_s: int) -> tuple[bool, int]:
        start, retry = _window(window_s, time.time())
        bucket = f"bu:rl:{key}:{start}"
        try:
            n = int(await self.redis.incr(bucket))
            if n == 1:
                await self.redis.expire(bucket, window_s + 1)
        except aioredis.RedisError as err:
            # Fail open: an outage of the limiter must not take the API down with it.
            log.warning("ratelimit_unavailable", error=str(err)[:200])
            return True, retry
        return n <= limit, retry
