from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.deps import validate_api_keys
from api.routes import debug, health, runs, secrets, sessions, usage
from api.services import ServicesFactory, build_services
from core.config import Settings, get_settings
from core.logging import configure_logging, get_logger


def create_app(
    settings: Settings | None = None, services_factory: ServicesFactory = build_services
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        s = settings or get_settings()
        configure_logging(s.log_level, json=s.log_json, redact=s.redact_pii)
        validate_api_keys(s)
        log = get_logger("api")
        async with services_factory(s) as services:
            app.state.services = services
            log.info("startup", env=s.env, tenants=len(set(s.api_keys.values())))
            yield
        log.info("shutdown")

    app = FastAPI(
        title="Browser Use Agent",
        version="0.2.0",
        description="Natural-language tasks executed by an LLM driving a real Chromium.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(runs.router)
    app.include_router(sessions.router)
    app.include_router(secrets.router)
    app.include_router(usage.router)
    app.include_router(debug.router)
    return app


app = create_app()
