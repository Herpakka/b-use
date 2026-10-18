"""Plain data the browser layer hands upward. No Playwright imports here: the agent uses these."""

from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import BaseModel, Field

MESSAGE_MAX = 200


class ElementInfo(BaseModel):
    index: int
    tag: str
    role: str | None = None
    name: str = ""
    type: str | None = None
    placeholder: str | None = None
    href: str | None = None
    value: str | None = None
    checked: bool | None = None
    disabled: bool = False
    options: list[str] | None = None
    frame: str | None = None
    in_viewport: bool = True


class ScrollState(BaseModel):
    above: int = 0
    below: int = 0
    viewport_height: int = 0


class Observation(BaseModel):
    dom_version: str
    url: str
    title: str
    elements: list[ElementInfo]
    # Page content in DOM order: an int is an element index, a str is visible text.
    outline: list[int | str]
    scroll: ScrollState
    tabs: int = 1
    dialogs: list[str] = Field(default_factory=list)
    blocked: list[str] = Field(default_factory=list)
    truncated: bool = False
    screenshot_b64: str | None = None

    def element(self, index: int) -> ElementInfo | None:
        return next((e for e in self.elements if e.index == index), None)

    def render(self, max_chars: int = 12_000) -> str:
        """Compact text form: ``[3]<button>Search</button>`` lines interleaved with page text."""
        head = [f"URL: {self.url}", f"Title: {self.title}"]
        if self.tabs > 1:
            head.append(f"Open tabs: {self.tabs} (you are on the newest one)")
        head += [f"Dialog auto-dismissed: {d}" for d in self.dialogs]
        head += [f"Blocked navigation (domain not allowed): {b}" for b in self.blocked]
        head.append(
            f"[scroll: {self.scroll.above}px above · {self.scroll.below}px below"
            f" · viewport {self.scroll.viewport_height}px]"
        )
        by_index = {e.index: e for e in self.elements}
        out = "\n".join(head)
        body: list[str] = []
        used = len(out)
        dropped = 0
        for i, item in enumerate(self.outline):
            line = render_element(by_index[item], self.url) if isinstance(item, int) else item
            if used + len(line) + 1 > max_chars:
                dropped = len(self.outline) - i
                break
            body.append(line)
            used += len(line) + 1
        if not self.elements:
            body.append("(no interactive elements in view)")
        if dropped or self.truncated:
            body.append("(observation truncated — scroll or use extract to see more)")
        return out + "\n" + "\n".join(body)


class ActionResult(BaseModel):
    ok: bool
    message: str
    # The target element changed since the observation the action was based on; nothing ran.
    stale: bool = False
    extracted_text: str | None = None
    url: str = ""


def short_url(url: str, base: str | None = None, limit: int = 80) -> str:
    if url.startswith("javascript:"):
        return "javascript:"
    parts = urlsplit(url)
    if base is not None and urlsplit(base).netloc == parts.netloc and parts.netloc:
        url = parts.path + (f"?{parts.query}" if parts.query else "")
    return url if len(url) <= limit else url[: limit - 1] + "…"


def render_element(e: ElementInfo, base_url: str | None = None) -> str:
    attrs: list[str] = []
    if e.type and e.tag in ("input", "button"):
        attrs.append(f"type={e.type}")
    if e.role and e.role != e.tag:
        attrs.append(f"role={e.role}")
    if e.placeholder:
        attrs.append(f'placeholder="{e.placeholder}"')
    if e.value is not None and e.tag in ("input", "textarea", "select"):
        attrs.append(f'value="{e.value}"')
    if e.checked is not None:
        attrs.append("checked" if e.checked else "unchecked")
    if e.disabled:
        attrs.append("disabled")
    if e.href:
        attrs.append(f'href="{short_url(e.href, base_url)}"')
    if e.options:
        attrs.append("options=" + "|".join(e.options))
    a = (" " + " ".join(attrs)) if attrs else ""
    prefix = f"[{e.index}]" + (f"(iframe {e.frame})" if e.frame else "")
    if e.tag == "input" or not e.name:
        return f"{prefix}<{e.tag}{a}>{e.name}"
    return f"{prefix}<{e.tag}{a}>{e.name}</{e.tag}>"


def clip_message(message: str) -> str:
    message = " ".join(message.split())
    return message if len(message) <= MESSAGE_MAX else message[: MESSAGE_MAX - 1] + "…"
