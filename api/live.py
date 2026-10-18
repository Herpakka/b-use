"""Live view: forward a session's CDP screencast to a WebSocket as binary JPEG frames."""

from __future__ import annotations

import asyncio
import contextlib

from starlette.websockets import WebSocket, WebSocketDisconnect

from browser.service import BrowserService, SessionNotFoundError

CLOSE_UNAUTHORIZED = 4401
CLOSE_NOT_FOUND = 4404
CLOSE_SESSION_GONE = 4410


async def stream_frames(
    ws: WebSocket, browser: BrowserService, sid: str, subprotocol: str | None
) -> None:
    await ws.accept(subprotocol=subprotocol)

    async def pump() -> None:
        async for frame in browser.screencast(sid):
            await ws.send_bytes(frame)

    async def until_disconnect() -> None:
        while (await ws.receive())["type"] != "websocket.disconnect":
            pass

    pump_task = asyncio.create_task(pump())
    recv_task = asyncio.create_task(until_disconnect())
    done, pending = await asyncio.wait({pump_task, recv_task}, return_when="FIRST_COMPLETED")
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    if pump_task in done:
        err = pump_task.exception()
        code, reason = (
            (CLOSE_SESSION_GONE, "browser session is not active")
            if err is None or isinstance(err, SessionNotFoundError)
            else (1011, "live view failed")
        )
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await ws.close(code=code, reason=reason)
