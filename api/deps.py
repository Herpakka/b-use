"""Request dependencies: services, API-key auth → tenant, per-tenant rate limiting."""

from __future__ import annotations

import hmac
import re
from typing import Annotated

from fastapi import Depends, HTTPException, Request, WebSocket
from starlette.datastructures import Headers

from api.services import Services
from core.config import Settings
from core.logging import bind_context

# Browsers cannot set headers on a WebSocket, so the key may travel as a subprotocol.
WS_KEY_PROTOCOL = "bu.key."
_TENANT = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def validate_api_keys(settings: Settings) -> None:
    for tenant in settings.api_keys.values():
        if not _TENANT.match(tenant):
            raise ValueError(f"invalid tenant name {tenant!r} in BU_API_KEYS")
    for key in settings.api_keys:
        if len(key) < 16:
            raise ValueError("API keys in BU_API_KEYS must be at least 16 characters")


def key_from_headers(headers: Headers) -> str | None:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return headers.get("x-api-key") or None


def tenant_for_key(settings: Settings, key: str | None) -> str | None:
    if not key:
        return None
    found = None
    for candidate, tenant in settings.api_keys.items():
        # Compare against every key so timing doesn't reveal which one matched.
        if hmac.compare_digest(candidate.encode(), key.encode()):
            found = tenant
    return found


async def require_tenant(
    request: Request, services: Annotated[Services, Depends(get_services)]
) -> str:
    settings = services.settings
    if not settings.api_keys:
        raise HTTPException(503, "no API keys are configured (BU_API_KEYS)")
    tenant = tenant_for_key(settings, key_from_headers(request.headers))
    if tenant is None:
        raise HTTPException(401, "missing or invalid API key", {"WWW-Authenticate": "Bearer"})
    allowed, retry_after = await services.limiter.hit(
        f"tenant:{tenant}", settings.rate_limit_per_minute, 60
    )
    if not allowed:
        raise HTTPException(429, "rate limit exceeded", {"Retry-After": str(retry_after)})
    bind_context(tenant=tenant)
    return tenant


def websocket_tenant(ws: WebSocket, settings: Settings) -> tuple[str, str | None] | None:
    """(tenant, subprotocol to echo) for a WebSocket handshake, or None if unauthenticated."""
    if key := key_from_headers(ws.headers):
        tenant = tenant_for_key(settings, key)
        return (tenant, None) if tenant else None
    for proto in ws.scope.get("subprotocols", []):
        if proto.startswith(WS_KEY_PROTOCOL):
            tenant = tenant_for_key(settings, proto[len(WS_KEY_PROTOCOL) :])
            return (tenant, proto) if tenant else None
    return None


ServicesDep = Annotated[Services, Depends(get_services)]
TenantDep = Annotated[str, Depends(require_tenant)]
