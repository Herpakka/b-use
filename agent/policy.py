"""Human gate for actions with lasting effects — a code policy, not a request in the prompt.

Whatever the model decides, a click or submit that looks like paying, ordering, sending,
deleting or confirming is routed through the ``human`` node first. False positives cost one
approval; false negatives can cost money, so the word list errs on the side of asking.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from agent.state import ElementBrief, PendingTool
from agent.tools import ToolParseError, parse_agent_tool
from browser.actions import Click, TypeText

_RISKY_LATIN = re.compile(
    r"\b(pay|payment|purchase|buy|checkout|check out|place order|order now|submit|delete|"
    r"remove|send|confirm|transfer|subscribe|unsubscribe|sign up|register|book now|"
    r"publish|post|donate|withdraw|cancel (?:my )?(?:order|subscription|account))\b",
    re.IGNORECASE,
)
# Thai has no word boundaries, so these match as substrings.
_RISKY_THAI = ("สั่งซื้อ", "ชำระ", "จ่ายเงิน", "ยืนยัน", "ลบ", "ส่ง", "โอน", "สมัคร", "จอง", "ซื้อ")
_SEARCH_LIKE = re.compile(r"\b(search|find|go|filter|look ?up)\b|ค้นหา", re.IGNORECASE)


def _risky(text: str) -> str | None:
    if m := _RISKY_LATIN.search(text):
        return m.group(0)
    return next((w for w in _RISKY_THAI if w in text), None)


def _search_like(el: ElementBrief) -> bool:
    return (
        el["type"] == "search"
        or el["role"] in ("search", "searchbox")
        or bool(_SEARCH_LIKE.search(el["name"]))
    )


def approval_reason(pending: PendingTool, elements: Mapping[str, ElementBrief]) -> str | None:
    """Why ``pending`` must be approved by a human first, or None."""
    try:
        action = parse_agent_tool(pending["name"], pending["args"])
    except ToolParseError:
        return None  # invalid calls never reach the browser
    if isinstance(action, Click):
        el = elements.get(str(action.index))
        if el is None:
            return None
        if word := _risky(el["name"]):
            return f'clicking "{el["name"]}" looks irreversible ({word})'
        if el["type"] == "submit" and not _search_like(el):
            return f'clicking submit button "{el["name"] or "submit"}" sends a form'
    if isinstance(action, TypeText) and action.submit:
        el = elements.get(str(action.index))
        if el is not None and not _search_like(el) and el["type"] != "password":
            return f'pressing Enter in "{el["name"] or el["tag"]}" may submit a form'
    return None
