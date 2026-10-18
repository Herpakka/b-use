"""DOM extraction: indexed interactive elements + visible text + scroll state, per frame.

The JS walks the *flat tree* (open shadow roots and slot assignments included), tags every
interactive element with ``data-bu-idx="<token>-<index>"`` and returns compact records. The
token changes on every observation, so stale attributes left on the page can never collide.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Frame, Page

from browser.models import ElementInfo, Observation, ScrollState, short_url

# Shared by the extractor and the freshness check so fingerprints are computed identically.
_HELPERS_JS = r"""
const clip = (s, n) => {
  s = (s || "").replace(/\s+/g, " ").trim();
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
};
const nameOf = (el, tag) => {
  const aria = el.getAttribute("aria-label");
  if (aria && aria.trim()) return clip(aria, 80);
  const lb = el.getAttribute("aria-labelledby");
  if (lb) {
    const root = el.getRootNode();
    const t = lb.split(/\s+/).map((id) => {
      const n = root.getElementById ? root.getElementById(id) : document.getElementById(id);
      return n ? n.textContent : "";
    }).join(" ");
    if (t.trim()) return clip(t, 80);
  }
  if (tag === "input" || tag === "textarea" || tag === "select") {
    const type = (el.type || "").toLowerCase();
    if (["submit", "button", "reset"].includes(type)) return clip(el.value || type, 80);
    if (el.labels && el.labels.length) return clip(el.labels[0].innerText, 80);
    return clip(el.getAttribute("title") || el.getAttribute("name") || "", 80);
  }
  let t = clip(el.innerText, 80);
  if (!t) {
    const img = el.querySelector("img[alt]");
    if (img) t = clip(img.getAttribute("alt"), 80);
  }
  if (!t) t = clip(el.getAttribute("title") || "", 80);
  return t;
};
const fingerprint = (el) => {
  const tag = el.localName;
  return tag + "|" + (el.getAttribute("role") || "") + "|" + nameOf(el, tag);
};
"""

EXTRACT_JS = (
    "(args) => {"
    + _HELPERS_JS
    + r"""
  const { token, start, expand, maxText, maxElements } = args;
  const SKIP = new Set(["script", "style", "noscript", "template", "head", "meta", "link",
    "svg", "iframe", "frame", "object", "embed", "canvas"]);
  const TAGS = new Set(["a", "button", "input", "select", "textarea", "summary"]);
  const ROLES = new Set(["button", "link", "checkbox", "radio", "tab", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "switch", "combobox", "textbox",
    "searchbox", "slider", "spinbutton", "treeitem"]);
  const vh = window.innerHeight, vw = window.innerWidth;
  const top = -expand, bottom = vh + expand;
  const items = [], elements = [];
  const visCache = new WeakMap();
  let textBudget = maxText, idx = start;

  const inRegion = (r) => r.width >= 1 && r.height >= 1 && r.bottom >= top && r.top <= bottom
    && r.right >= 0 && r.left <= vw;
  const shown = (el) => {
    let v = visCache.get(el);
    if (v === undefined) {
      v = typeof el.checkVisibility === "function"
        ? el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })
        : getComputedStyle(el).visibility !== "hidden";
      visCache.set(el, v);
    }
    return v;
  };
  const strictInteractive = (el, tag) => {
    if (tag === "a") return el.hasAttribute("href");
    if (tag === "input") return (el.type || "").toLowerCase() !== "hidden";
    if (TAGS.has(tag)) return true;
    const role = el.getAttribute("role");
    if (role && ROLES.has(role)) return true;
    if (el.hasAttribute("onclick")) return true;
    if (el.isContentEditable && !(el.parentElement && el.parentElement.isContentEditable)) return true;
    const ti = el.getAttribute("tabindex");
    return ti !== null && parseInt(ti, 10) >= 0;
  };
  // Click handlers attached with addEventListener are invisible to the DOM; a pointer cursor
  // that its parent doesn't share is the best cheap signal.
  const pointerHint = (el, tag) => {
    if (tag === "body" || tag === "html" || tag === "label") return false;
    if (getComputedStyle(el).cursor !== "pointer") return false;
    const root = el.getRootNode();
    const p = el.parentElement || (root && root.host);
    return !(p && getComputedStyle(p).cursor === "pointer");
  };
  // Text runs are merged into one line for compactness, but never across a block boundary
  // (heading, paragraph, list item, row …): the model needs to tell "the first paragraph" apart.
  const BLOCKS = new Set(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "tr",
    "blockquote", "pre", "figcaption", "caption", "section", "article", "header", "footer",
    "aside", "nav", "main", "form", "fieldset", "legend", "table", "ul", "ol", "dl", "br", "hr"]);
  let lineBreak = false;
  const pushText = (t) => {
    if (textBudget <= 0) return;
    t = clip(t, Math.min(400, textBudget));
    if (!t) return;
    textBudget -= t.length;
    const last = items[items.length - 1];
    if (!lineBreak && last && last[0] === "t" && last[1].length + t.length < 400) last[1] += " " + t;
    else items.push(["t", t]);
    lineBreak = false;
  };
  // display:contents boxes report as invisible although their children render.
  const isContents = (el) => getComputedStyle(el).display === "contents";
  const textVisible = (node) => {
    const p = node.parentElement;
    if (!p || (!shown(p) && !isContents(p))) return false;
    const range = document.createRange();
    range.selectNodeContents(node);
    return inRegion(range.getBoundingClientRect());
  };
  // Sensitive fields and the kind of form an element belongs to, for the approval policy.
  const PAY = /(card.?(num|no\b)|cc.?num|credit.?card|cvv|cvc|csc|security.?code|expir|exp.?(date|month|year)|iban|routing.?num|account.?num|sort.?code|หมายเลขบัตร|บัตรเครดิต)/i;
  const fieldKind = (el) => {
    const tag = el.localName;
    if (tag !== "input" && tag !== "select" && tag !== "textarea") return null;
    const ac = (el.getAttribute("autocomplete") || "").toLowerCase();
    if (/(^|\s)cc-/.test(ac)) return "payment";
    if ((el.type || "").toLowerCase() === "password" || ac.includes("password")) return "password";
    const hint = [el.name, el.id, el.getAttribute("placeholder"), el.getAttribute("aria-label")].join(" ");
    return PAY.test(hint) ? "payment" : null;
  };
  const formCache = new WeakMap();
  const formKind = (el) => {
    const form = el.form || el.closest("form");
    if (!form) return null;
    let kind = formCache.get(form);
    if (kind === undefined) {
      const fields = Array.from(form.elements || []);
      const kinds = fields.map(fieldKind);
      kind = kinds.includes("payment") ? "payment"
        : kinds.includes("password") ? "login"
        : form.getAttribute("role") === "search" || fields.some((f) => (f.type || "").toLowerCase() === "search") ? "search"
        : "form";
      formCache.set(form, kind);
    }
    return kind;
  };
  const record = (el, tag, rect) => {
    const index = idx++;
    if (token) el.setAttribute("data-bu-idx", token + "-" + index);
    const type = tag === "input" ? (el.type || "text").toLowerCase() : el.getAttribute("type");
    const e = {
      index, tag, role: el.getAttribute("role"), name: nameOf(el, tag), type,
      placeholder: el.getAttribute("placeholder"), href: null, value: null, checked: null,
      disabled: !!el.disabled || el.getAttribute("aria-disabled") === "true", options: null,
      in_viewport: rect.bottom >= 0 && rect.top <= vh, fp: fingerprint(el),
      field: fieldKind(el), form: formKind(el),
    };
    if (tag === "a") e.href = el.href || el.getAttribute("href");
    if (tag === "input" || tag === "textarea") {
      if (type === "checkbox" || type === "radio") e.checked = !!el.checked;
      else if (!["submit", "button", "reset", "file", "image"].includes(type)) {
        e.value = type === "password" ? (el.value ? "••••" : "") : clip(el.value, 60);
      }
    }
    if (tag === "select") {
      const sel = el.selectedOptions && el.selectedOptions[0];
      e.value = sel ? clip(sel.textContent, 60) : "";
      e.options = Array.from(el.options).slice(0, 15).map((o) => clip(o.textContent, 40));
    }
    const ac = el.getAttribute("aria-checked");
    if (ac === "true" || ac === "false") e.checked = ac === "true";
    elements.push(e);
    items.push(["e", index]);
  };
  // Meaning carried only by attributes: image alt text, aria-labels on icons, rating classes
  // ("star-rating Four" renders as icon glyphs with no text at all).
  const RATING = /(^|[\s_-])(rating|stars?)([\s_-]|$)/i;
  const annotate = (el, tag) => {
    let note = null;
    if (tag === "img") note = el.getAttribute("alt");
    else {
      const aria = el.getAttribute("aria-label")
        || (el.getAttribute("role") === "img" ? el.getAttribute("title") : null);
      const cls = typeof el.className === "string" ? el.className : "";
      const candidate = aria || (RATING.test(cls) ? cls : null);
      if (candidate && !(el.textContent || "").trim()) note = candidate;
    }
    if (!note || !note.trim()) return;
    if (inRegion(el.getBoundingClientRect()) && shown(el)) {
      pushText("[" + (tag === "img" ? "image: " : "") + clip(note, 80) + "]");
    }
  };
  const walk = (node, inside) => {
    if (node.nodeType === Node.TEXT_NODE) {
      if (!inside && node.textContent.trim() && textVisible(node)) pushText(node.textContent);
      return;
    }
    if (node.nodeType === Node.DOCUMENT_FRAGMENT_NODE) {
      for (const c of node.childNodes) walk(c, inside);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const el = node, tag = el.localName;
    if (SKIP.has(tag)) return;
    if (tag === "slot") {
      const assigned = el.assignedNodes({ flatten: true });
      for (const c of (assigned.length ? assigned : el.childNodes)) walk(c, inside);
      return;
    }
    // display:none / content-visibility hides the whole subtree.
    if (typeof el.checkVisibility === "function" && !el.checkVisibility() && !isContents(el)) return;
    let interactive = false;
    if (elements.length < maxElements) {
      const strict = strictInteractive(el, tag);
      if (strict || !inside) {
        const rect = el.getBoundingClientRect();
        if (inRegion(rect) && (strict || pointerHint(el, tag)) && shown(el)) {
          record(el, tag, rect);
          interactive = true;
        }
      }
    }
    if (!interactive && !inside) annotate(el, tag);
    const block = BLOCKS.has(tag);
    if (block) lineBreak = true;
    const kids = el.shadowRoot ? el.shadowRoot.childNodes : el.childNodes;
    for (const c of kids) walk(c, inside || interactive);
    if (block) lineBreak = true;
  };
  walk(document.body || document.documentElement, false);
  const se = document.scrollingElement || document.documentElement;
  return {
    url: location.href, title: document.title, items, elements,
    truncated: elements.length >= maxElements,
    scroll: { y: Math.round(window.scrollY), height: Math.round(se.scrollHeight), vh: Math.round(vh) },
  };
}"""
)

FINGERPRINT_JS = "(el) => {" + _HELPERS_JS + " return fingerprint(el); }"

_WHOLE_PAGE_PX = 10_000_000


@dataclass(frozen=True)
class ElementRef:
    """Where an observed element lives, so the executor can find and re-verify it."""

    frame: Frame
    selector: str
    fingerprint: str
    tag: str
    name: str
    type: str | None


async def _frame_on_screen(frame: Frame, viewport_height: int, expansion_px: int) -> bool:
    try:
        handle = await frame.frame_element()
        box = await handle.bounding_box()
    except PlaywrightError:
        return False
    if box is None or box["width"] < 2 or box["height"] < 2:
        return False
    return box["y"] < viewport_height + expansion_px and box["y"] + box["height"] > -expansion_px


async def build_observation(
    page: Page,
    *,
    expansion_px: int,
    max_elements: int,
    max_text: int = 6000,
    frame_allowed: Callable[[str], bool] | None = None,
) -> tuple[Observation, dict[int, ElementRef]]:
    """``frame_allowed`` filters child frames by URL: a frame redirected off the allow-list is
    never shown to the model."""
    token = secrets.token_hex(4)
    viewport_height = (page.viewport_size or {"height": 800})["height"]
    outline: list[int | str] = []
    elements: list[ElementInfo] = []
    refs: dict[int, ElementRef] = {}
    next_index = 1
    title, url, scroll, truncated = "", page.url, ScrollState(), False

    main = page.main_frame
    frames = [main, *(f for f in page.frames if f is not main)]
    for frame in frames:
        is_main = frame is main
        if not is_main and frame_allowed is not None and not frame_allowed(frame.url):
            continue
        if not is_main and not await _frame_on_screen(frame, viewport_height, expansion_px):
            continue
        remaining = max_elements - len(elements)
        if remaining <= 0:
            truncated = True
            break
        try:
            data: dict[str, Any] = await frame.evaluate(
                EXTRACT_JS,
                {
                    "token": token,
                    "start": next_index,
                    "expand": expansion_px,
                    "maxText": max_text if is_main else max_text // 4,
                    "maxElements": remaining,
                },
            )
        except PlaywrightError:
            continue  # frame navigated or detached mid-extraction
        label = None if is_main else short_url(frame.url, limit=60)
        if label and (data["elements"] or data["items"]):
            outline.append(f"--- iframe {label} ---")
        for kind, value in data["items"]:
            outline.append(int(value) if kind == "e" else str(value))
        for raw in data["elements"]:
            fp = raw.pop("fp")
            info = ElementInfo.model_validate({**raw, "frame": label})
            elements.append(info)
            refs[info.index] = ElementRef(
                frame=frame,
                selector=f'[data-bu-idx="{token}-{info.index}"]',
                fingerprint=fp,
                tag=info.tag,
                name=info.name,
                type=info.type,
            )
        next_index += len(data["elements"])
        truncated = truncated or bool(data["truncated"])
        if is_main:
            title, url = str(data["title"]), str(data["url"])
            s = data["scroll"]
            scroll = ScrollState(
                above=int(s["y"]),
                below=max(0, int(s["height"]) - int(s["y"]) - int(s["vh"])),
                viewport_height=int(s["vh"]),
            )

    observation = Observation(
        dom_version=token,
        url=url,
        title=title,
        elements=elements,
        outline=outline,
        scroll=scroll,
        tabs=len(page.context.pages),
        truncated=truncated,
    )
    return observation, refs


def _extract_line(e: dict[str, Any]) -> str:
    name = str(e.get("name") or "")
    if e["tag"] == "select":
        return f'[select "{name}": selected "{e.get("value") or ""}"]'
    if e.get("checked") is not None:
        state = "checked" if e["checked"] else "unchecked"
        return f'[{e.get("type") or "checkbox"} "{name}": {state}]'
    if e["tag"] in ("input", "textarea") and e.get("value") is not None:
        label = name or str(e.get("placeholder") or "")
        return f'[field "{label}": "{e["value"]}"]'
    return name


async def page_text(
    page: Page, max_chars: int, frame_allowed: Callable[[str], bool] | None = None
) -> str:
    """Whole-page text for Extract, in reading order.

    Unlike ``innerText`` it includes form state (selected options, typed values, checkboxes),
    image alt text and attribute-only labels such as rating classes. ``frame_allowed`` filters
    child frames exactly as in ``build_observation``.
    """
    viewport_height = (page.viewport_size or {"height": 800})["height"]
    main = page.main_frame
    chunks: list[str] = []
    for frame in [main, *(f for f in page.frames if f is not main)]:
        if frame is not main and frame_allowed is not None and not frame_allowed(frame.url):
            continue
        if frame is not main and not await _frame_on_screen(frame, viewport_height, _WHOLE_PAGE_PX):
            continue
        try:
            data: dict[str, Any] = await frame.evaluate(
                EXTRACT_JS,
                {
                    "token": None,  # read-only: leave the observation's data-bu-idx tags alone
                    "start": 1,
                    "expand": _WHOLE_PAGE_PX,
                    "maxText": max_chars,
                    "maxElements": 2000,
                },
            )
        except PlaywrightError:
            continue
        by_index = {e["index"]: e for e in data["elements"]}
        lines = [
            str(value) if kind == "t" else _extract_line(by_index[value])
            for kind, value in data["items"]
        ]
        if text := "\n".join(line for line in lines if line):
            chunks.append(text)
    return "\n\n".join(chunks)[:max_chars]
