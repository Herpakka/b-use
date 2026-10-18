"""Server-sent events for a run: replay from the store, then the live tail from the bus."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable

from api.manager import FINISHED
from api.services import Services
from core.bus import run_channel
from core.store import TERMINAL, RunEvent

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # nginx and friends must not buffer the stream.
    "X-Accel-Buffering": "no",
}


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
    live tail (dropped bus message) is filled from the store.
    """
    store = services.store
    async with services.bus.subscribe(run_channel(run_id)) as sub:
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
        while not await is_disconnected():
            message = await sub.get(services.settings.sse_heartbeat_s)
            if message is None:
                yield ": ping\n\n"
                continue
            event = RunEvent.model_validate_json(message)
            if event.seq <= last:
                continue
            missed = await store.events_after(run_id, last) if event.seq > last + 1 else [event]
            for e in missed:
                yield format_sse(e)
                last = e.seq
                if e.type == FINISHED:
                    return
