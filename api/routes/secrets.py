from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Response

from api.deps import ServicesDep, TenantDep
from api.schemas import SecretIn
from core.secrets import MIN_VALUE_LENGTH, SECRET_NAME, SecretInfo, SecretsUnavailableError

router = APIRouter(prefix="/v1/secrets", tags=["secrets"])

SecretName = Annotated[str, Path(pattern=SECRET_NAME)]
MAX_VALUE_LENGTH = 4096


@router.get("", response_model=list[SecretInfo])
async def list_secrets(tenant: TenantDep, services: ServicesDep) -> list[SecretInfo]:
    """Names and domains of this tenant's secrets. Values are never returned."""
    return await services.secrets.list(tenant)


@router.put("/{name}", status_code=204, response_class=Response)
async def put_secret(
    name: SecretName, body: SecretIn, tenant: TenantDep, services: ServicesDep
) -> Response:
    """Store (or replace) a secret. Refer to it in a task as ``{{SECRET:<name>}}``; the agent
    types it without seeing it, and only on ``domains`` if given."""
    value = body.value.get_secret_value()
    # Checked here rather than by the model so a 422 never echoes the value back.
    if not MIN_VALUE_LENGTH <= len(value) <= MAX_VALUE_LENGTH:
        raise HTTPException(
            422, f"value must be {MIN_VALUE_LENGTH}–{MAX_VALUE_LENGTH} characters long"
        )
    try:
        await services.secrets.put(tenant, name, value, body.domains)
    except SecretsUnavailableError as err:
        raise HTTPException(503, str(err)) from err
    return Response(status_code=204)


@router.delete("/{name}", status_code=204, response_class=Response)
async def delete_secret(name: SecretName, tenant: TenantDep, services: ServicesDep) -> Response:
    if not await services.secrets.delete(tenant, name):
        raise HTTPException(404, "secret not found")
    return Response(status_code=204)
