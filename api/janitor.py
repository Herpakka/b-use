"""Retention: what is kept, and for how long.

* Screenshots: ``BU_SCREENSHOT_TTL_S`` (24 h by default).
* Agent checkpoints of finished runs — they hold page snapshots and step history:
  ``BU_CHECKPOINT_TTL_S`` after the run ends (24 h). A finished run is never resumed.
* Whole runs with their events: ``BU_RUN_RETENTION_DAYS`` after they end (off by default).

Browser cookies of a run's own (unnamed) session are deleted when the run ends; named
sessions keep theirs, since that is what they are for.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from core.config import Settings
from core.logging import get_logger
from core.store import RunStore, utcnow

log = get_logger("api.janitor")

PURGE_BATCH = 200


class Janitor:
    def __init__(
        self, store: RunStore, checkpointer: BaseCheckpointSaver[Any] | None, settings: Settings
    ) -> None:
        self.store = store
        self.checkpointer = checkpointer
        self.settings = settings
        self._task: asyncio.Task[None] | None = None

    async def sweep(self, now: datetime | None = None) -> dict[str, int]:
        now = now or utcnow()
        s = self.settings
        done = {
            "screenshots": await self.store.delete_artifacts_before(
                now - timedelta(seconds=s.screenshot_ttl_s)
            ),
            "checkpoints": 0,
            "runs": 0,
        }
        cutoff = now - timedelta(seconds=s.checkpoint_ttl_s)
        while run_ids := await self.store.runs_to_purge(cutoff, PURGE_BATCH):
            for run_id in run_ids:
                if self.checkpointer is not None:
                    await self.checkpointer.adelete_thread(run_id)
                await self.store.update_run(run_id, purged_at=now)
            done["checkpoints"] += len(run_ids)
            if len(run_ids) < PURGE_BATCH:
                break
        if s.run_retention_days is not None:
            done["runs"] = await self.store.delete_runs_before(
                now - timedelta(days=s.run_retention_days)
            )
        if any(done.values()):
            log.info("retention_sweep", **done)
        return done

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="janitor")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            try:
                await self.sweep()
            except Exception as err:  # retention must keep running
                log.error("retention_sweep_failed", error=str(err)[:200])
            await asyncio.sleep(self.settings.janitor_interval_s)
