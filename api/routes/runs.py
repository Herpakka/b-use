from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, WebSocket
from fastapi.responses import StreamingResponse

from api.deps import ServicesDep, TenantDep, websocket_tenant
from api.events import SSE_HEADERS, run_stream
from api.live import CLOSE_NOT_FOUND, CLOSE_UNAUTHORIZED, stream_frames
from api.manager import RunError
from api.schemas import CreateRunRequest, EventOut, ResumeRequest, RunOut
from api.services import Services
from browser.service import SessionNotFoundError
from core.store import TERMINAL, RunRecord

router = APIRouter(prefix="/v1/runs", tags=["runs"])


async def _run(services: Services, tenant: str, run_id: str) -> RunRecord:
    run = await services.store.get_run(run_id, tenant)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


def _raise(err: RunError) -> HTTPException:
    return HTTPException(err.status, err.detail)


@router.post("", status_code=202, response_model=RunOut)
async def create_run(
    body: CreateRunRequest, tenant: TenantDep, services: ServicesDep, response: Response
) -> RunOut:
    """Start a run. It executes in the background; follow it with ``/stream``."""
    try:
        run = await services.manager.create(tenant, body)
    except RunError as err:
        raise _raise(err) from err
    response.headers["Location"] = f"/v1/runs/{run.id}"
    return RunOut.of(run)


@router.get("", response_model=list[RunOut])
async def list_runs(
    tenant: TenantDep, services: ServicesDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[RunOut]:
    return [RunOut.of(r) for r in await services.store.list_runs(tenant, limit)]


@router.get("/{run_id}", response_model=RunOut)
async def get_run(run_id: str, tenant: TenantDep, services: ServicesDep) -> RunOut:
    return RunOut.of(await _run(services, tenant, run_id))


@router.get("/{run_id}/events", response_model=list[EventOut])
async def list_events(
    run_id: str,
    tenant: TenantDep,
    services: ServicesDep,
    after: Annotated[int, Query(ge=0)] = 0,
) -> list[EventOut]:
    await _run(services, tenant, run_id)
    return [EventOut.of(e) for e in await services.store.events_after(run_id, after)]


@router.get(
    "/{run_id}/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_run(
    run_id: str,
    request: Request,
    tenant: TenantDep,
    services: ServicesDep,
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    """Server-sent events, one per step. Reconnect with ``Last-Event-ID`` to continue."""
    await _run(services, tenant, run_id)
    start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
    return StreamingResponse(
        run_stream(services, run_id, start, request.is_disconnected),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/{run_id}/resume", status_code=202, response_model=RunOut)
async def resume_run(
    run_id: str, body: ResumeRequest, tenant: TenantDep, services: ServicesDep
) -> RunOut:
    """Answer the agent while the run is ``needs_human``: approve/deny or reply."""
    try:
        run = await services.manager.resume(tenant, run_id, body)
    except RunError as err:
        raise _raise(err) from err
    return RunOut.of(run)


@router.post("/{run_id}/cancel", response_model=RunOut)
async def cancel_run(run_id: str, tenant: TenantDep, services: ServicesDep) -> RunOut:
    try:
        run = await services.manager.cancel(tenant, run_id)
    except RunError as err:
        raise _raise(err) from err
    return RunOut.of(run)


@router.get(
    "/{run_id}/screenshot",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "no screenshot yet"}},
)
async def screenshot(run_id: str, tenant: TenantDep, services: ServicesDep) -> Response:
    """The live page while the session is open, else the last one captured."""
    run = await _run(services, tenant, run_id)
    headers = {"Cache-Control": "no-store"}
    if run.status not in TERMINAL:
        try:
            image = await services.browser.screenshot(run.session_id)
            return Response(image, media_type="image/jpeg", headers=headers)
        except SessionNotFoundError:
            pass
    stored = await services.store.get_artifact(run_id, "screenshot")
    if stored is None:
        raise HTTPException(404, "no screenshot yet")
    content, content_type = stored
    return Response(content, media_type=content_type, headers=headers)


@router.websocket("/{run_id}/live")
async def run_live(ws: WebSocket, run_id: str) -> None:
    """Binary JPEG frames of the run's browser while its session is open."""
    services: Services = ws.app.state.services
    auth = websocket_tenant(ws, services.settings)
    if auth is None:
        await ws.close(code=CLOSE_UNAUTHORIZED)
        return
    tenant, subprotocol = auth
    run = await services.store.get_run(run_id, tenant)
    if run is None:
        await ws.close(code=CLOSE_NOT_FOUND)
        return
    await stream_frames(ws, services.browser, run.session_id, subprotocol)
