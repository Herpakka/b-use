"""Mask personal data in logs and stored run events.

Only what is *kept* is masked: log lines and the event log behind SSE. The agent itself still
sees pages as they are, and a run's task/result/pending question stay intact in its record
(the client asked for them; retention removes them later — see ``api.janitor``).

Covered: e-mail addresses, payment card numbers (Luhn-checked), Thai national ID numbers
(checksum), phone numbers (9–15 digits written like one) and common API key/token shapes.
Patterns err towards masking: an ISBN may come out as ``[phone]``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

_KEY = re.compile(
    r"\b(?:sk|pk|rk)-(?:proj-|live-|test-)?[A-Za-z0-9_-]{16,}"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\bxox[abpr]-[A-Za-z0-9-]{10,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_DIGIT_RUN = re.compile(r"(?<![\w.,])\+?\(?\d[\d ().-]{7,22}\d(?![\w])")
_DATE = re.compile(r"\d{4}[-.]\d{1,2}[-.]\d{1,2}\b")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _thai_id(digits: str) -> bool:
    if len(digits) != 13:
        return False
    s = sum(int(d) * (13 - i) for i, d in enumerate(digits[:12]))
    return (11 - s % 11) % 10 == int(digits[12])


def _number(match: re.Match[str]) -> str:
    text = match.group(0)
    digits = _digits(text)
    separated = bool(re.search(r"\d[ .-]\d", text))
    if 13 <= len(digits) <= 19 and _luhn(digits) and (separated or text.isdigit()):
        return "[card]"
    if _thai_id(digits):
        return "[national-id]"
    if _DATE.match(text):
        return text  # "2026-10-18 14:05" is not a phone number
    if 9 <= len(digits) <= 15 and (text.startswith(("+", "0", "(")) or separated):
        return "[phone]"
    return text


_RULES: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str] | str]] = [
    (_KEY, "[key]"),
    (_EMAIL, "[email]"),
    (_DIGIT_RUN, _number),
]


def redact_text(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def redact(value: Any) -> Any:
    """``value`` with every string inside it (dicts, lists, tuples) masked."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(redact(v) for v in value)
    return value


def structlog_processor(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> Mapping[str, Any]:
    """structlog processor: mask every value of a log line (keys and the event name stay)."""
    return {
        k: (v if k in ("event", "timestamp", "level", "logger") else redact(v))
        for k, v in event_dict.items()
    }
