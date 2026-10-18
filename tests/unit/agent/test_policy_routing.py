from typing import Any

import pytest

from agent.policy import approval_reason
from agent.routing import is_looping, limit_reason, route_after_act, route_after_decide
from agent.state import AgentState, ElementBrief, Limits, PendingTool, new_state


def el(
    name: str,
    tag: str = "button",
    type: str | None = None,
    *,
    field: str | None = None,
    form: str | None = None,
) -> ElementBrief:
    return ElementBrief(tag=tag, name=name, type=type, role=None, field=field, form=form)


ELEMENTS = {
    "1": el("Add to cart"),
    "2": el("Place order"),
    "3": el("Search", type="submit"),
    "4": el("Continue", type="submit"),
    "5": el("ยืนยันการชำระเงิน"),
    "6": el("Delete account", tag="a"),
    "7": el("Search the catalogue", tag="input", type="search"),
    "8": el("Comment", tag="textarea"),
    "9": el("Password", tag="input", type="password"),
    "10": el("Senders list", tag="a"),
}


def click(i: int) -> PendingTool:
    return PendingTool(name="Click", args={"index": i})


@pytest.mark.parametrize("index", [2, 4, 5, 6])
def test_risky_clicks_need_approval(index: int) -> None:
    assert approval_reason(click(index), ELEMENTS) is not None


@pytest.mark.parametrize("index", [1, 3, 10, 99])
def test_safe_clicks_pass(index: int) -> None:
    assert approval_reason(click(index), ELEMENTS) is None


def test_enter_in_search_box_passes_but_other_submits_need_approval() -> None:
    def typed(i: int) -> PendingTool:
        return PendingTool(name="TypeText", args={"index": i, "text": "x", "submit": True})

    assert approval_reason(typed(7), ELEMENTS) is None
    assert approval_reason(typed(9), ELEMENTS) is None
    assert approval_reason(typed(8), ELEMENTS) is not None
    no_submit = PendingTool(name="TypeText", args={"index": 8, "text": "x"})
    assert approval_reason(no_submit, ELEMENTS) is None


CONTEXT = {
    "1": el("Continue", type="button", form="payment"),
    "2": el("Card number", tag="input", type="text", field="payment", form="payment"),
    "3": el("Apply coupon", tag="a", form="payment"),
    "4": el("Complete", type="button"),
    "5": el("Sign in", type="submit", form="login"),
    "6": el("Save", type="submit", form="login"),  # e.g. a change-password form
    "7": el("Email", tag="input", type="email", form="login"),
    "8": el("Next", tag="a"),
    "9": el("Go", type="submit", form="search"),
}


def typed(i: int, text: str = "x", submit: bool = False) -> PendingTool:
    return PendingTool(name="TypeText", args={"index": i, "text": text, "submit": submit})


def test_context_gates_buttons_without_risky_words() -> None:
    checkout = "https://shop.test/checkout/step-3"
    assert "payment form" in str(approval_reason(click(1), CONTEXT))
    assert "payment form" in str(approval_reason(click(3), CONTEXT))
    assert approval_reason(click(4), CONTEXT, "https://shop.test/product/7") is None
    assert "checkout page" in str(approval_reason(click(4), CONTEXT, checkout))
    assert approval_reason(click(8), CONTEXT, checkout) is None  # a link, not a button
    assert approval_reason(click(9), CONTEXT, checkout) is None  # searching
    assert "Complete order" in str(
        approval_reason(click(1), {"1": el("Complete order")}, "https://x.test/")
    )


def test_logging_in_passes_other_password_forms_do_not() -> None:
    assert approval_reason(click(5), CONTEXT) is None
    assert approval_reason(click(6), CONTEXT) is not None
    assert approval_reason(typed(7, submit=True), CONTEXT) is None


def test_payment_fields_need_approval_unless_a_stored_secret_is_typed() -> None:
    assert "payment field" in str(approval_reason(typed(2, "4111 1111 1111 1111"), CONTEXT))
    assert approval_reason(typed(2, "{{SECRET:card}}"), CONTEXT) is None
    reason = approval_reason(typed(2, "{{SECRET:card}}", submit=True), CONTEXT)
    assert "payment form" in str(reason)


def test_old_checkpoints_without_form_context_still_work() -> None:
    legacy = {"1": ElementBrief(tag="button", name="Continue", type="submit", role=None)}
    assert approval_reason(click(1), legacy) is not None


def test_invalid_or_non_click_calls_are_not_gated() -> None:
    assert approval_reason(PendingTool(name="Click", args={"index": "x"}), ELEMENTS) is None
    assert (
        approval_reason(PendingTool(name="Navigate", args={"url": "https://a"}), ELEMENTS) is None
    )


def state(**kw: Any) -> AgentState:
    s = new_state(
        run_id="r",
        task="t",
        allowed_domains=["a.test"],
        limits=Limits(max_steps=5, budget_usd=0.1, run_timeout_s=60),
        prompt_version="v",
        now=0.0,
    )
    s.update(kw)  # type: ignore[typeddict-item]
    return s


def test_limit_reasons() -> None:
    assert limit_reason(state()) is None
    assert "max_steps" in (limit_reason(state(step=5)) or "")
    assert "budget" in (limit_reason(state(cost_usd=0.1)) or "")
    assert limit_reason(state(), now=59) is None
    assert "timeout" in (limit_reason(state(), now=60) or "")


def test_is_looping() -> None:
    assert not is_looping(["a", "a"])
    assert not is_looping(["a", "b", "a"])
    assert is_looping(["b", "a", "a", "a"])
    assert is_looping(["a", "b", "a", "b", "a"])  # alternating cycle
    assert not is_looping(["a", "x", "y", "z", "w", "v", "a", "a"])  # outside the 6-step window


def test_route_after_decide() -> None:
    assert route_after_decide(state(pending=click(1))) == "act"
    assert route_after_decide(state(pending=click(2), status="needs_human")) == "human"
    assert route_after_decide(state(pending=PendingTool(name="Done", args={}))) == "finalize"
    assert route_after_decide(state(status="failed")) == "finalize"


def test_route_after_act() -> None:
    assert route_after_act(state(last_result={"ok": True})) == "observe"
    assert route_after_act(state(last_result={"stale": True})) == "observe"
    assert route_after_act(state(errors_in_row=1)) == "reflect"
    assert route_after_act(state(signatures=["s", "s", "s"])) == "reflect"
    assert route_after_act(state(errors_in_row=1, reflections_in_row=4)) == "finalize"
    assert route_after_act(state(step=5)) == "finalize"
