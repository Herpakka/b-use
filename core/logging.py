"""structlog JSON logging with run/session ids carried through contextvars."""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor


def configure_logging(level: str = "INFO", *, json: bool = True) -> None:
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(),
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial: Any) -> FilteringBoundLogger:
    return cast(FilteringBoundLogger, structlog.get_logger(name, **initial))


def bind_context(**values: Any) -> None:
    """Attach e.g. ``run_id`` / ``session_id`` to every log line in the current task."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
