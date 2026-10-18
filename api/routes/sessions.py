from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Response, WebSocket

from api.deps import ServicesDep, TenantDep, websocket_tenant
from api.live import CLOSE_NOT_FOUND, CLOSE_UNAUTHORIZED, stream_frames
from api.schemas import SESSION_NAME, SessionOut, session_id, session_name
from api.services import Services

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

_NAME = re.compile(SESSION_NAME)
_RUN_SESSION = re.compile(r"^run-[0-9a-f]{12}$")


def _sid(tenant: str, name: str) -> str:
    if not (_NAME.match(name) or _RUN_SESSION.match(name)):
        raise HTTPException(404, "session not found")
    return session_id(tenant, name)


@router.get("", response_model=list[SessionOut])
async def list_sessions(tenant: TenantDep, services: ServicesDep) -> list[SessionOut]:
    """Open browser sessions of this tenant (closed ones keep their cookies on disk)."""
    prefix = session_id(tenant, "")
    return [
        SessionOut(
            name=session_name(tenant, s.id),
            url=s.url,
            tabs=s.tabs,
            idle_s=s.idle_s,
            busy=s.busy,
            held=s.held,
        )
        for s in await services.browser.list_sessions()
        if s.id.startswith(prefix)
    ]


@router.delete("/{name}", status_code=204, response_class=Response)
async def close_session(name: str, tenant: TenantDep, services: ServicesDep) -> Response:
    sid = _sid(tenant, name)
    if services.manager.session_busy(sid):
        raise HTTPException(409, "a run is using this session; cancel it first")
    if sid not in {s.id for s in await services.browser.list_sessions()}:
        raise HTTPException(404, "session not found")
    await services.browser.close(sid)
    return Response(status_code=204)


@router.websocket("/{name}/live")
async def session_live(ws: WebSocket, name: str) -> None:
    services: Services = ws.app.state.services
    auth = websocket_tenant(ws, services.settings)
    if auth is None:
        await ws.close(code=CLOSE_UNAUTHORIZED)
        return
    tenant, subprotocol = auth
    if not (_NAME.match(name) or _RUN_SESSION.match(name)):
        await ws.close(code=CLOSE_NOT_FOUND)
        return
    await stream_frames(ws, services.browser, session_id(tenant, name), subprotocol)
