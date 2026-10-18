from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.context import AgentContext
from agent.state import AgentState, ElementBrief
from browser.actions import Navigate
from browser.service import SessionNotFoundError

RESTORE_ATTEMPTS = 3


async def observe_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Snapshot the current page. Re-creates the browser session if it is gone (restart,
    reaper), restoring cookies from storage and returning to the last known URL."""
    ctx = runtime.context
    sid = state["session_id"]
    update: dict[str, Any] = {}
    for attempt in range(1, RESTORE_ATTEMPTS + 1):
        try:
            if not await ctx.browser.has_session(sid):
                await ctx.browser.open(
                    sid, allowed_domains=state["allowed_domains"], vision=state["vision"]
                )
                target = state["current_url"] or state["start_url"]
                if target and target != "about:blank":
                    result = await ctx.browser.execute(sid, Navigate(url=target))
                    if not result.ok:
                        update["history"] = [f"(session restored) {result.message}"]
            obs = await ctx.browser.observe(sid)
            break
        except SessionNotFoundError:
            # The browser went away mid-way (crash, restart): open the session again.
            if attempt == RESTORE_ATTEMPTS:
                raise
    elements: dict[str, ElementBrief] = {
        str(e.index): ElementBrief(
            tag=e.tag, name=e.name, type=e.type, role=e.role, field=e.field, form=e.form
        )
        for e in obs.elements
    }
    return {
        **update,
        "observation": obs.render(ctx.settings.observation_max_chars),
        "dom_version": obs.dom_version,
        "current_url": obs.url,
        "elements": elements,
    }
