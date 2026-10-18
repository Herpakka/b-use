"""Tracing with Langfuse: a trace per graph invocation (a run, or its continuation after a human
answers), grouped into one Langfuse *session* per run; a span per node; a generation per LLM call
with tokens and cost. On when both Langfuse keys are set.

What is sent — prompts, i.e. page content — goes through the same PII masking as the logs when
``BU_REDACT_PII`` is on. The secret values never appear in prompts in the first place.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.config import Settings
from core.logging import get_logger
from core.redact import redact

log = get_logger("agent.tracing")

TRACE_NAME = "browser-run"


class Tracer:
    def __init__(self, settings: Settings) -> None:
        # Imported here: a deployment without Langfuse never loads its OpenTelemetry stack.
        from langfuse import Langfuse

        assert settings.langfuse_public_key and settings.langfuse_secret_key
        self._redact = settings.redact_pii
        self.client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_host,  # wins over the SDK's own LANGFUSE_BASE_URL
            environment=settings.env,
            mask=self._mask,
        )
        log.info("tracing_enabled", host=settings.langfuse_host)

    def _mask(self, *, data: Any, **_: Any) -> Any:
        return redact(data) if self._redact else data

    def config(
        self, run_id: str, *, user: str | None = None, tags: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Extra RunnableConfig for one graph invocation."""
        from langfuse.langchain import CallbackHandler

        metadata: dict[str, Any] = {"langfuse_session_id": run_id, "langfuse_tags": list(tags)}
        if user:
            metadata["langfuse_user_id"] = user
        return {"callbacks": [CallbackHandler()], "run_name": TRACE_NAME, "metadata": metadata}

    def flush(self) -> None:
        self.client.flush()

    def shutdown(self) -> None:
        self.client.shutdown()


def make_tracer(settings: Settings) -> Tracer | None:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        return Tracer(settings)
    except Exception as err:  # tracing must never stop runs
        log.warning("tracing_unavailable", error=str(err)[:200])
        return None
