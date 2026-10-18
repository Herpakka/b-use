from __future__ import annotations

from fastapi import APIRouter

from api.deps import ServicesDep, TenantDep
from api.manager import month_start
from api.schemas import UsageOut

router = APIRouter(prefix="/v1/usage", tags=["usage"])


@router.get("", response_model=UsageOut)
async def usage(tenant: TenantDep, services: ServicesDep) -> UsageOut:
    """This tenant's spend in the current calendar month (UTC) and what is left under the caps."""
    since = month_start()
    used = await services.store.usage(since, tenant)
    return UsageOut(
        month=since.strftime("%Y-%m"),
        runs=used.runs,
        cost_usd=round(used.cost_usd, 6),
        committed_usd=round(used.committed_usd, 6),
        tokens_in=used.tokens_in,
        tokens_out=used.tokens_out,
        budget_usd=services.settings.tenant_monthly_budget_usd,
        remaining_usd=await services.manager.remaining_budget(tenant),
    )
