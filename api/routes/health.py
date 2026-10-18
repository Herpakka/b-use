from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import metrics
from api.services import Services

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process answers. Restart it if this fails."""
    return {"status": "ok"}


async def _check(probe: Callable[[], Awaitable[None]]) -> str:
    try:
        await asyncio.wait_for(probe(), 3)
    except Exception as err:
        return f"{type(err).__name__}: {str(err)[:120]}".rstrip(": ")
    return "ok"


@router.get("/metrics", include_in_schema=False)
async def prometheus(request: Request) -> Response:
    """Prometheus metrics of this replica (not authenticated: keep it off the public proxy)."""
    services: Services = request.app.state.services
    browser = await services.browser.health()
    metrics.BROWSER_UP.set(1 if browser.ok else 0)
    metrics.BROWSER_SESSIONS.set(browser.sessions)
    metrics.BROWSER_CAPACITY.set(browser.max_sessions)
    metrics.BROWSER_RESTARTS.set(browser.restarts)
    manager = services.manager
    metrics.RUNS_ACTIVE.labels("running").set(manager.running)
    metrics.RUNS_ACTIVE.labels("paused").set(max(0, manager.owned - manager.running))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/readyz", responses={503: {"description": "a dependency is down"}})
async def readyz(request: Request) -> JSONResponse:
    """Readiness: runs can be served — database, event bus, browser and OpenAI key all OK.
    Take the instance out of rotation (don't restart it) while this fails."""
    services: Services = request.app.state.services
    browser = await services.browser.health()
    checks: dict[str, Any] = {
        "database": await _check(services.store.ping),
        "bus": await _check(services.bus.ping),
        "browser": browser.model_dump(),
        "llm": "ok" if services.manager.ctx is not None else "BU_OPENAI_API_KEY is not set",
    }
    checks["draining"] = services.manager.draining
    checks["replica"] = services.manager.replica
    ok = (
        browser.ok
        and not services.manager.draining
        and all(checks[k] == "ok" for k in ("database", "bus", "llm"))
    )
    return JSONResponse(
        {"status": "ready" if ok else "unavailable", "checks": checks},
        status_code=200 if ok else 503,
    )
