"""Human gate for actions with lasting effects — a code policy, not a request in the prompt.

Whatever the model decides, these go through the ``human`` node first:

* clicks/submits whose label says pay, order, send, delete, confirm, … (any language we know);
* anything that submits a form, except searches and logging in;
* any click inside a payment form, and a "Continue"/"Next"-style button on a checkout page
  — so "Complete" or "Continue" on the last step of a purchase is caught without a keyword;
* typing into a payment field (unless it is a ``{{SECRET:…}}`` placeholder the user stored).

False positives cost one approval; false negatives can cost money, so it errs on asking.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlsplit

from agent.state import ElementBrief, PendingTool
from agent.tools import ToolParseError, parse_agent_tool
from browser.actions import Click, TypeText
from core.secrets import PLACEHOLDER

_RISKY_LATIN = re.compile(
    r"\b(pay|payment|purchase|buy|checkout|check out|place (?:my |your )?order|order now|"
    r"complete (?:the |my |your )?(?:order|purchase|payment|booking|checkout)|submit|delete|"
    r"remove|send|confirm|transfer|subscribe|unsubscribe|sign up|register|book now|apply now|"
    r"create (?:an |my |your )?account|(?:close|deactivate) (?:my |your )?account|"
    r"publish|post|donate|withdraw|cancel (?:my )?(?:order|subscription|account))\b",
    re.IGNORECASE,
)
# Thai has no word boundaries, so these match as substrings.
_RISKY_THAI = ("สั่งซื้อ", "ชำระ", "จ่ายเงิน", "ยืนยัน", "ลบ", "ส่ง", "โอน", "สมัคร", "จอง", "ซื้อ")
_SEARCH_LIKE = re.compile(r"\b(search|find|go|filter|look ?up)\b|ค้นหา", re.IGNORECASE)
_LOGIN_LIKE = re.compile(r"\b(log ?in|sign ?in|next|continue)\b|เข้าสู่ระบบ|ถัดไป", re.IGNORECASE)
# Buttons that move a multi-step flow forward: harmless on a product page, the final step on a
# checkout page.
_COMMIT_LIKE = re.compile(
    r"\b(continue|next|proceed|complete|finish|place|ok|yes|done|review)\b"
    r"|ดำเนินการต่อ|ถัดไป|ตกลง|เสร็จ",
    re.IGNORECASE,
)
_CHECKOUT_PATH = re.compile(
    r"(checkout|payment|billing|purchase|/pay(?:/|$|\b)|/order|/cart|/basket|confirm)",
    re.IGNORECASE,
)


def _risky(text: str) -> str | None:
    if m := _RISKY_LATIN.search(text):
        return m.group(0)
    return next((w for w in _RISKY_THAI if w in text), None)


def _search_like(el: ElementBrief) -> bool:
    return (
        el["type"] == "search"
        or el["role"] in ("search", "searchbox")
        or el.get("form") == "search"
        or bool(_SEARCH_LIKE.search(el["name"]))
    )


def _button_like(el: ElementBrief) -> bool:
    return el["tag"] == "button" or el["role"] == "button" or el["type"] in ("submit", "button")


def _checkout_page(url: str | None) -> bool:
    return bool(url) and bool(_CHECKOUT_PATH.search(urlsplit(url or "").path))


def approval_reason(
    pending: PendingTool, elements: Mapping[str, ElementBrief], url: str | None = None
) -> str | None:
    """Why ``pending`` must be approved by a human first, or None. ``url`` is the page's."""
    try:
        action = parse_agent_tool(pending["name"], pending["args"])
    except ToolParseError:
        return None  # invalid calls never reach the browser
    if not isinstance(action, Click | TypeText):
        return None
    el = elements.get(str(action.index))
    if el is None:
        return None
    name, form = el["name"], el.get("form")

    if isinstance(action, Click):
        if word := _risky(name):
            return f'clicking "{name}" looks irreversible ({word})'
        if form == "payment" and not _search_like(el):
            return f'clicking "{name or el["tag"]}" in a payment form'
        if _button_like(el) and _COMMIT_LIKE.search(name) and _checkout_page(url):
            return f'clicking "{name}" on a checkout page may complete a purchase'
        if el["type"] == "submit" and not _search_like(el):
            if form == "login" and _LOGIN_LIKE.search(name):
                return None  # signing in is not a lasting change
            return f'clicking submit button "{name or "submit"}" sends a form'
        return None

    if el.get("field") == "payment" and not PLACEHOLDER.fullmatch(action.text.strip()):
        return f'typing into payment field "{name or el["tag"]}"'
    if action.submit and not _search_like(el) and el["type"] != "password" and form != "login":
        if form == "payment":
            return "pressing Enter submits a payment form"
        return f'pressing Enter in "{name or el["tag"]}" may submit a form'
    return None
