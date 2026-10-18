"""End-to-end graph behaviour with a scripted LLM and a fake browser (no network, no Chromium)."""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.runner import drive
from browser.actions import Click, Extract, Navigate, Scroll
from browser.models import ActionResult
from tests.agent_fakes import (
    PLAN,
    FakeBrowser,
    StepClock,
    decide_prompts,
    make_ctx,
    page,
    run_task,
    text,
    tool,
)

HOME = "https://shop.test/"
SHOP = {HOME: page(HOME, ("a", "Kettles"), ("button", "Place order"), body="Welcome")}


def clicked(message: str, url: str = HOME, ok: bool = True) -> ActionResult:
    return ActionResult(ok=ok, message=message, url=url)


async def test_happy_path() -> None:
    browser = FakeBrowser(SHOP, results=[clicked('click [1] "Kettles" → ok', HOME + "kettles")])
    ctx = make_ctx(browser, [PLAN, tool("Click", index=1), tool("Done", result="7 kettles")])
    final, pending, _ = await run_task(ctx)

    assert pending is None
    assert final["status"] == "done" and final["result"] == "7 kettles"
    assert final["step"] == 2
    assert final["history"] == ['1. click [1] "Kettles" → ok', "2. done → 7 kettles"]
    assert browser.executed == [Navigate(url=HOME), Click(index=1)]
    assert browser.forgotten == [final["session_id"]]  # its own session: cookies not kept
    assert final["plan"] == ["open the page", "find the answer"]
    # 3 decide-model calls × (1000 in · $2/M + 50 out · $8/M)
    assert final["cost_usd"] == pytest.approx(3 * (0.002 + 0.0004))
    assert final["node_costs"] == pytest.approx({"plan": 0.0024, "decide": 2 * 0.0024})
    assert final["tokens_in"] == 3000

    first_decision = decide_prompts(ctx)[1]
    assert "# Task\nfind the answer" in first_decision
    assert '">\nURL: https://shop.test/' in first_decision
    assert "[1]<a>Kettles</a>" in first_decision


async def test_max_steps_stops_the_run() -> None:
    results = [clicked(f"scroll → position {i}") for i in range(5)]
    browser = FakeBrowser(SHOP, results=results)
    script = [PLAN] + [tool("Scroll", direction="down") for _ in range(5)]
    ctx = make_ctx(browser, script, max_steps=3)
    final, _, _ = await run_task(ctx)

    assert final["status"] == "failed"
    assert "max_steps (3)" in final["failure_reason"]
    assert browser.executed[1:] == [Scroll(direction="down")] * 3


async def test_budget_stops_the_run() -> None:
    browser = FakeBrowser(SHOP)
    # plan + one decision cost 0.0048 > 0.004
    ctx = make_ctx(browser, [PLAN, tool("Scroll"), tool("Scroll")], budget_usd=0.004)
    final, _, _ = await run_task(ctx)
    assert final["status"] == "failed" and "budget" in final["failure_reason"]
    assert final["step"] == 1


async def test_timeout_stops_the_run() -> None:
    results = [clicked(f"scroll → position {i}") for i in range(5)]
    browser = FakeBrowser(SHOP, results=results)
    ctx = make_ctx(
        browser,
        [PLAN] + [tool("Scroll") for _ in range(5)],
        clock=StepClock(step=60),
        run_timeout_s=150,
    )
    final, _, _ = await run_task(ctx)
    assert final["status"] == "failed" and "timeout" in final["failure_reason"]


async def test_repeated_action_triggers_reflection() -> None:
    same = 'click [1] "Kettles" → ok'
    browser = FakeBrowser(SHOP, results=[clicked(same) for _ in range(3)])
    reflection = tool(
        "Reflection",
        diagnosis="Clicking Kettles does nothing.",
        revised_plan=["use search"],
        advice="Use the search box instead.",
    )
    script = [PLAN, *[tool("Click", index=1)] * 3, reflection, tool("Done", result="x")]
    ctx = make_ctx(browser, script)
    final, _, _ = await run_task(ctx)

    assert "↺ reflected: Clicking Kettles does nothing." in final["history"]
    assert final["plan"] == ["use search"]
    assert "# Advice\nUse the search box instead." in decide_prompts(ctx)[-1]
    assert final["status"] == "done"


async def test_error_triggers_reflection() -> None:
    browser = FakeBrowser(SHOP, results=[clicked("click [1] → error: timed out", ok=False)])
    reflection = tool("Reflection", diagnosis="Overlay.", revised_plan=["x"], advice="Close it.")
    ctx = make_ctx(browser, [PLAN, tool("Click", index=1), reflection, tool("Done", result="x")])
    final, _, _ = await run_task(ctx)
    assert final["history"][:2] == ["1. click [1] → error: timed out", "↺ reflected: Overlay."]


async def test_stale_observation_does_not_consume_a_step() -> None:
    stale = ActionResult(ok=False, stale=True, message="[1] changed — re-observing")
    browser = FakeBrowser(SHOP, results=[stale, clicked('click [1] "Kettles" → ok')])
    script = [PLAN, tool("Click", index=1), tool("Click", index=1), tool("Done", result="x")]
    ctx = make_ctx(browser, script)
    final, _, _ = await run_task(ctx)

    assert browser.executed[1:] == [Click(index=1), Click(index=1)]
    assert final["step"] == 2  # one counted click + done
    assert final["history"][0] == '1. click [1] "Kettles" → ok'


async def test_invalid_tool_call_is_an_error_step() -> None:
    browser = FakeBrowser(SHOP)
    reflection = tool("Reflection", diagnosis="Bad args.", revised_plan=["x"], advice="Use ints.")
    script = [PLAN, tool("Click", index="first"), reflection, tool("Done", result="x")]
    ctx = make_ctx(browser, script)
    final, _, _ = await run_task(ctx)
    assert final["history"][0].startswith("1. invalid call Click → error: index:")
    assert browser.executed == [Navigate(url=HOME)]  # nothing reached the browser


async def test_extract_is_answered_by_the_small_model() -> None:
    browser = FakeBrowser(SHOP, page_text="Blue Kettle — Price: ฿1,290")
    ctx = make_ctx(
        browser,
        [PLAN, tool("Extract", query="price"), tool("Done", result="฿1,290")],
        [text("฿1,290")],
    )
    final, _, _ = await run_task(ctx)

    assert browser.executed[1] == Extract(query="price")
    assert final["findings"] == ["price (https://shop.test/): ฿1,290"]
    assert final["history"][0] == '1. extract "price" → ฿1,290'
    small_prompt = str(ctx.llm_small.calls[0][-1].content)  # type: ignore[attr-defined]
    assert '">\nBlue Kettle — Price: ฿1,290\n</page_content boundary="' in small_prompt
    # gpt-4.1-mini: 1000 · $0.4/M + 50 · $1.6/M on top of 3 decide calls
    assert final["cost_usd"] == pytest.approx(3 * 0.0024 + 0.0004 + 0.00008)


async def test_irreversible_click_needs_approval_and_runs_when_approved() -> None:
    seen: list[dict[str, Any]] = []

    async def approve(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return {"approved": True}

    browser = FakeBrowser(SHOP, results=[clicked('click [2] "Place order" → ok')])
    ctx = make_ctx(browser, [PLAN, tool("Click", index=2), tool("Done", result="ordered")])
    final, _, _ = await run_task(ctx, on_human=approve)

    assert seen[0]["kind"] == "approval"
    assert seen[0]["action"] == 'Click [2] "Place order"'
    assert "place order" in seen[0]["reason"].lower()
    assert Click(index=2) in browser.executed
    assert 'human approved: Click [2] "Place order"' in final["history"]
    assert final["status"] == "done"


async def test_irreversible_click_is_skipped_when_denied() -> None:
    async def deny(payload: dict[str, Any]) -> dict[str, Any]:
        return {"approved": False, "answer": "too expensive"}

    browser = FakeBrowser(SHOP)
    script = [PLAN, tool("Click", index=2), tool("Done", result="not ordered", success=False)]
    ctx = make_ctx(browser, script)
    final, _, _ = await run_task(ctx, on_human=deny)

    assert Click(index=2) not in browser.executed
    assert 'human denied: Click [2] "Place order" — "too expensive"' in final["history"]
    assert "The human refused" in decide_prompts(ctx)[-1]
    assert final["status"] == "failed"


async def test_ask_human_question_and_answer() -> None:
    async def answer(payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["kind"] == "question" and payload["question"] == "Which size?"
        return {"approved": True, "answer": "Large"}

    browser = FakeBrowser(SHOP)
    script = [PLAN, tool("AskHuman", question="Which size?"), tool("Done", result="Large")]
    ctx = make_ctx(browser, script)
    final, _, _ = await run_task(ctx, on_human=answer)

    assert 'asked human "Which size?" → "Large"' in final["history"]
    assert "The human answered: Large" in decide_prompts(ctx)[-1]
    assert final["status"] == "done"


async def test_paused_run_survives_a_restart() -> None:
    saver = InMemorySaver()
    browser = FakeBrowser(SHOP)
    ctx = make_ctx(browser, [PLAN, tool("Click", index=2)])
    state, pending, _ = await run_task(ctx, saver=saver, run_id="run-restart")
    assert pending is not None and pending["kind"] == "approval"
    assert state["status"] == "needs_human"
    assert browser.held == {state["session_id"]}  # kept alive while the human decides

    # "Restart": a fresh browser with no sessions and fresh models, same checkpoint store.
    browser2 = FakeBrowser(SHOP)
    ctx2 = make_ctx(browser2, [tool("Done", result="resumed")])

    async def approve(payload: dict[str, Any]) -> dict[str, Any]:
        return {"approved": True}

    from agent.graph import compile_graph

    final, _ = await drive(
        compile_graph(saver), ctx2, "run-restart", None, max_steps=25, on_human=approve
    )
    # The approved click referred to a page from the dead browser: it is re-observed, not replayed.
    assert browser2.executed == [Navigate(url=HOME)]
    assert final["status"] == "done" and final["result"] == "resumed"


async def test_alternating_cycle_triggers_reflection() -> None:
    results = [clicked('type "Option 2" into [1] → ok'), clicked('extract "state" → read')] * 3
    browser = FakeBrowser(SHOP, results=results)
    reflection = tool("Reflection", diagnosis="Cycling.", revised_plan=["x"], advice="Read value=.")
    pair = [tool("TypeText", index=1, text="Option 2"), tool("Scroll")]
    script = [PLAN, *pair, *pair, pair[0], reflection, tool("Done", result="x")]
    ctx = make_ctx(browser, script)
    final, _, _ = await run_task(ctx)
    assert final["history"][5] == "↺ reflected: Cycling."  # after A,B,A,B,A


async def test_extract_not_found_is_an_error_step() -> None:
    browser = FakeBrowser(SHOP, page_text="nothing useful")
    reflection = tool(
        "Reflection", diagnosis="Not on page.", revised_plan=["x"], advice="Look elsewhere."
    )
    ctx = make_ctx(
        browser,
        [
            PLAN,
            tool("Extract", query="rating"),
            reflection,
            tool("Done", result="?", success=False),
        ],
        [text("NOT FOUND — only prices")],
    )
    final, _, _ = await run_task(ctx)
    assert final["history"][:2] == [
        '1. extract "rating" → NOT FOUND — only prices',
        "↺ reflected: Not on page.",
    ]
    assert final["findings"] == []  # NOT FOUND is not a finding


async def test_session_is_held_only_while_the_human_decides() -> None:
    held_during: list[set[str]] = []
    browser = FakeBrowser(SHOP, results=[clicked('click [2] "Place order" → ok')])

    async def approve(payload: dict[str, Any]) -> dict[str, Any]:
        held_during.append(set(browser.held))
        return {"approved": True}

    ctx = make_ctx(browser, [PLAN, tool("Click", index=2), tool("Done", result="ok")])
    final, _, _ = await run_task(ctx, on_human=approve)
    assert held_during == [{final["session_id"]}]
    assert browser.held == set()
