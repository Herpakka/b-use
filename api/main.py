from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import health
from core.config import get_settings
from core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    get_logger("api").info("startup", env=settings.env)
    # Phase 3: start BrowserPool + reaper here and stop them on shutdown.
    yield
    get_logger("api").info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(title="Browser Use Agent", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    return app


app = create_app()
