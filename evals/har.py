"""Pin real websites for evals with HAR files (phase 5.2).

``record``: run live and save every response of each task to ``<dir>/<task-id>.har.zip``.
``replay``: serve those responses instead of the network, so a failing task means the agent
changed, not the website. Requests the recording doesn't have go to the network (the agent may
take a different path than when it was recorded); ``replay-strict`` refuses them instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from browser.local import LocalBrowserService
from browser.pool import BrowserPool

HarMode = Literal["record", "replay", "replay-strict"]


class HarBrowserService(LocalBrowserService):
    def __init__(self, pool: BrowserPool, mode: HarMode, directory: Path) -> None:
        super().__init__(pool)
        self.mode = mode
        self.directory = directory
        self.task_id = "default"  # set by the runner before each task

    def path(self, task_id: str) -> Path:
        return self.directory / f"{task_id}.har.zip"

    async def open(
        self, session_id: str, *, allowed_domains: Sequence[str], vision: bool = False
    ) -> None:
        await super().open(session_id, allowed_domains=allowed_domains, vision=vision)
        session = self.pool.get(session_id)
        assert session is not None
        har = self.path(self.task_id)
        if self.mode == "record":
            self.directory.mkdir(parents=True, exist_ok=True)
            # Written when the context closes, i.e. when the run ends.
            await session.context.route_from_har(
                har, update=True, update_content="attach", update_mode="minimal"
            )
        elif har.exists():
            # Registered after the network guard, so it answers first; what it does not have
            # falls through to the guard (and the network) unless strict.
            strict = self.mode == "replay-strict"
            await session.context.route_from_har(har, not_found="abort" if strict else "fallback")
