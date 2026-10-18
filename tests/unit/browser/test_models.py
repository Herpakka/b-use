from browser.actions import ActionAdapter, Click, Done, TypeText
from browser.models import (
    ElementInfo,
    Observation,
    ScrollState,
    clip_message,
    render_element,
    short_url,
)


def _obs(**kw: object) -> Observation:
    base: dict[str, object] = {
        "dom_version": "abc",
        "url": "https://shop.test/p",
        "title": "Shop",
        "elements": [
            ElementInfo(index=1, tag="a", name="Home", href="https://shop.test/"),
            ElementInfo(index=2, tag="input", type="search", placeholder="Search", value=""),
            ElementInfo(index=3, tag="button", name="Add to cart"),
        ],
        "outline": ["Welcome", 1, 2, "Price ฿10", 3],
        "scroll": ScrollState(above=0, below=1200, viewport_height=800),
    }
    base.update(kw)
    return Observation.model_validate(base)


def test_render_interleaves_text_and_elements() -> None:
    text = _obs().render()
    assert text.splitlines() == [
        "URL: https://shop.test/p",
        "Title: Shop",
        "[scroll: 0px above · 1200px below · viewport 800px]",
        "Welcome",
        '[1]<a href="/">Home</a>',
        '[2]<input type=search placeholder="Search" value="">',
        "Price ฿10",
        "[3]<button>Add to cart</button>",
    ]


def test_render_truncates_in_order() -> None:
    text = _obs().render(max_chars=120)
    assert "[3]" not in text
    assert text.endswith("(observation truncated — scroll or use extract to see more)")


def test_render_reports_dialogs_blocked_and_tabs() -> None:
    text = _obs(tabs=2, dialogs=["alert: hi"], blocked=["https://x.test/ (nope)"]).render()
    assert "Open tabs: 2" in text
    assert "Dialog auto-dismissed: alert: hi" in text
    assert "Blocked by network policy: https://x.test/ (nope)" in text


def test_render_element_variants() -> None:
    sel = ElementInfo(index=4, tag="select", name="Size", value="M", options=["S", "M"])
    assert render_element(sel) == '[4]<select value="M" options=S|M>Size</select>'
    box = ElementInfo(index=5, tag="input", type="checkbox", checked=False, name="Agree")
    assert render_element(box) == "[5]<input type=checkbox unchecked>Agree"
    framed = ElementInfo(index=6, tag="button", name="Pay", frame="pay.test/x", disabled=True)
    assert render_element(framed) == "[6](iframe pay.test/x)<button disabled>Pay</button>"


def test_short_url() -> None:
    assert short_url("https://a.test/x?y=1", "https://a.test/") == "/x?y=1"
    assert short_url("https://b.test/x", "https://a.test/") == "https://b.test/x"
    assert short_url("javascript:void(0)") == "javascript:"
    assert len(short_url("https://a.test/" + "x" * 200)) == 80


def test_clip_message() -> None:
    assert clip_message("a\n  b") == "a b"
    assert len(clip_message("x" * 500)) == 200


def test_action_adapter_round_trip() -> None:
    assert ActionAdapter.validate_python({"kind": "click", "index": 3}) == Click(index=3)
    t = TypeText(index=2, text="hi", submit=True)
    assert ActionAdapter.validate_python(t.model_dump()) == t
    assert ActionAdapter.validate_python({"kind": "done", "result": "x"}) == Done(result="x")
