import pytest

from core.redact import redact, redact_text, structlog_processor


def thai_id(first12: str) -> str:
    s = sum(int(d) * (13 - i) for i, d in enumerate(first12))
    return first12 + str((11 - s % 11) % 10)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail alice.b+shop@example.co.th now", "mail [email] now"),
        ("card 4111 1111 1111 1111 ok", "card [card] ok"),
        ("card 4111111111111111", "card [card]"),
        ("call 081-234-5678 or +66 81 234 5678", "call [phone] or [phone]"),
        ("office (02) 123 4567", "office [phone]"),
        ("us 555-123-4567", "us [phone]"),
        (f"id {thai_id('110370207181')}", "id [national-id]"),
        ("key sk-proj-abcdefghijklmnopqrstuv123", "key [key]"),
        ("Authorization: Bearer abcdefghijklmnop.qrstuvwxyz", "Authorization: [key]"),
        ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "token [key]"),
    ],
)
def test_personal_data_is_masked(text: str, expected: str) -> None:
    assert redact_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Blue Kettle costs ฿1,290 (was 1,590.00)",
        "on 2026-10-18 14:26:42 the run took 17.3 s",
        "run-9c50748bcdd9 · session acme.run-9c50748bcdd9",
        "https://books.toscrape.com/catalogue/page-2.html?id=1234567890",
        "card 4111 1111 1111 1112",  # fails the Luhn check
        "1234567890123",  # 13 digits, wrong ID checksum, no separators
        "step 12 of 25, 3 tabs, 250 elements",
    ],
)
def test_ordinary_text_is_left_alone(text: str) -> None:
    assert redact_text(text) == text


def test_nested_values_and_log_lines() -> None:
    data = {"task": "email bob@example.com", "steps": [{"line": "typed 0812345678"}], "n": 3}
    assert redact(data) == {"task": "email [email]", "steps": [{"line": "typed [phone]"}], "n": 3}
    line = structlog_processor(None, "info", {"event": "run_crashed", "error": "x@y.com failed"})
    assert line == {"event": "run_crashed", "error": "[email] failed"}
