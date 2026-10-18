"""Memory: one context stays within budget, 50 actions don't grow it unboundedly, and closing
the context gives the memory back (what matters for a pool that runs for days)."""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from browser.actions import AnyAction, Navigate, Scroll, TypeText
from browser.pool import BrowserPool
from tests.browser_utils import FIXTURE_DOMAINS, find, goto, serve_fixtures
from tests.conftest import make_settings

pytestmark = pytest.mark.browser

MB = 1024 * 1024
CONTEXT_BUDGET_MB = 80
LEAK_BUDGET_MB = 30
CLOSE_RESIDUE_MB = 20


def _is_chromium(proc: psutil.Process) -> bool:
    name = proc.name().lower()
    return "chrom" in name or "headless" in name


def _chromium_procs() -> list[psutil.Process]:
    procs = []
    for proc in psutil.Process(os.getpid()).children(recursive=True):
        try:
            if _is_chromium(proc):
                procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


def tree_uss(roots: list[psutil.Process]) -> int:
    """Unique memory of the given browser processes and all their children."""
    total = 0
    for root in roots:
        try:
            for proc in [root, *root.children(recursive=True)]:
                total += proc.memory_full_info().uss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


async def test_fifty_actions_no_leak(tmp_path: Path) -> None:
    # A browser of its own, so other tests' contexts can't skew the numbers.
    existing = {p.pid for p in _chromium_procs()}
    pool = BrowserPool(make_settings(tmp_path))
    try:
        await pool.start()
    except Exception as err:
        pytest.skip(f"Chromium unavailable: {err}")
    roots = [
        p
        for p in _chromium_procs()
        if p.pid not in existing and not ((parent := p.parent()) and _is_chromium(parent))
    ]
    try:
        chromium_uss: Callable[[], int] = lambda: tree_uss(roots)  # noqa: E731
        before_context = chromium_uss()
        sid = f"mem-{secrets.token_hex(3)}"
        s = await pool.open_session(sid, allowed_domains=FIXTURE_DOMAINS)
        await serve_fixtures(s.context)
        try:
            await goto(s, "static.html")
            per_context = chromium_uss() - before_context

            async def run(i: int) -> int:
                """One cycle of 5 actions; the type target is resolved after the navigation."""
                actions: list[AnyAction] = [
                    Navigate(url="https://fixtures.test/static.html"),
                    TypeText(index=1, text=f"item {i}"),
                    Scroll(direction="down", pages=1),
                    Scroll(direction="up", pages=1),
                    Navigate(url="https://fixtures.test/spa.html"),
                ]
                for action in actions:
                    if isinstance(action, TypeText):
                        obs = await s.observe()
                        action = action.model_copy(update={"index": find(obs, "Search products")})
                    result = await s.execute(action)
                    assert result.ok, result.message
                return len(actions)

            for i in range(5):  # warm-up: let caches, bfcache and the JIT fill before the baseline
                await run(i)
            baseline = chromium_uss()

            done, i = 0, 0
            while done < 50:
                done += await run(i)
                i += 1
            growth = chromium_uss() - baseline
        finally:
            await pool.close_session(sid, persist=False)
        await asyncio.sleep(1.0)  # let the renderer process exit
        residue = chromium_uss() - before_context
    finally:
        await pool.stop()

    print(
        f"\nper-context {per_context / MB:.1f} MB · 50-action growth {growth / MB:.1f} MB"
        f" · after close {residue / MB:+.1f} MB"
    )
    assert per_context < CONTEXT_BUDGET_MB * MB
    assert growth < LEAK_BUDGET_MB * MB
    assert residue < CLOSE_RESIDUE_MB * MB
