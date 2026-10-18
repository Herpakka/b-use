"""A one-page debug UI: start runs, watch steps and the live browser, answer the agent."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from api.deps import ServicesDep

router = APIRouter(include_in_schema=False)


@lru_cache
def _page() -> str:
    return files("api").joinpath("static/debug.html").read_text(encoding="utf-8")


@router.get("/")
async def root(services: ServicesDep) -> RedirectResponse:
    return RedirectResponse("/debug" if services.settings.debug_page else "/docs")


@router.get("/debug", response_class=HTMLResponse)
async def debug_page(services: ServicesDep) -> HTMLResponse:
    if not services.settings.debug_page:
        raise HTTPException(404)
    # The page holds no data: everything it shows comes from API calls made with the key the
    # operator types in, so serving it unauthenticated is fine.
    return HTMLResponse(
        _page(),
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline';"
                " script-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:"
            ),
        },
    )
