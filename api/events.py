"""Server-sent events for a run: replay from the store, then the live tail from the bus."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from api.manager import FINISHED
from api.services import Services
from core.bus import Subscription, run_channel
from core.logging import get_logger
from core.store import TERMINAL, RunEvent

log = get_logger("api.events")

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # nginx and friends must not buffer the stream.
    "X-Accel-Buffering": "no",
}
# Without the bus (Redis down) the stream polls the store this often instead.
POLL_S = 1.0


def format_sse(event: RunEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"


async def run_stream(
    services: Services,
    run_id: str,
    after: int,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[str]:
    """Yield SSE frames from event ``after + 1`` on; ends after the ``finished`` event.

    Subscribes before replaying so nothing published in between is missed; any gap in the
    live tail (dropped bus message) is filled from the store. If the bus is unavailable the
    stream still works, polling the store once a second.
    """
    store = services.store
    async with contextlib.AsyncExitStack() as stack:
        sub: Subscription | None
        try:
            sub = await stack.enter_async_context(services.bus.subscribe(run_channel(run_id)))
        except Exception as err:
            log.warning("sse_without_bus", run_id=run_id, error=str(err)[:200])
            sub = None
        last = after
        for event in await store.events_after(run_id, last):
            yield format_sse(event)
            last = event.seq
            if event.type == FINISHED:
                return
        run = await store.get_run(run_id)
        if run is None or run.status in TERMINAL:
            return
        yield ": connected\n\n"
        idle = 0.0
        while not await is_disconnected():
            if sub is None:
                await asyncio.sleep(POLL_S)
                fresh = await store.events_after(run_id, last)
                idle = 0.0 if fresh else idle + POLL_S
                if idle >= services.settings.sse_heartbeat_s:
                    idle = 0.0
                    yield ": ping\n\n"
            else:
                message = await sub.get(services.settings.sse_heartbeat_s)
                if message is None:
                    yield ": ping\n\n"
                    continue
                event = RunEvent.model_validate_json(message)
                if event.seq <= last:
                    continue
                gap = event.seq > last + 1
                fresh = await store.events_after(run_id, last) if gap else [event]
            for e in fresh:
                yield format_sse(e)
                last = e.seq
                if e.type == FINISHED:
                    return
