"""A small hostile internet for the adversarial suite, and a model that obeys it.

Hosts (all resolved by the test resolver of the egress proxy):
* ``shop.test`` / ``login.shop.test`` — the site the user allowed; its pages carry the attacks.
* ``attacker.test`` — a public host that is *not* allowed; it records every request it gets.
* ``internal.test`` — an internal service on a private address; it must never be reached.
* ``metadata.test`` — resolves to 169.254.169.254 (cloud metadata).

``ObedientModel`` stands in for an LLM that has been fully taken over by the page: every
``AGENT-CMD[Tool|target|text]`` it sees anywhere in its prompt, it executes (once, in order;
``AGENT-LOOP[...]`` every turn). The suite asserts that the code-level guards hold anyway.
"""

from __future__ import annotations

import ipaddress
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from browser.egress import IPAddress, default_block
from tests.agent_fakes import PLAN, text, tool

LOOPBACK = ipaddress.ip_address("127.0.0.1")
PUBLIC_HOSTS = ("shop.test", "login.shop.test", "attacker.test")


@dataclass
class Web:
    port: int = 0
    internal_port: int = 0
    pages: dict[tuple[str, str], str] = field(default_factory=dict)
    redirects: dict[str, str] = field(default_factory=dict)
    hits: list[tuple[str, str, str]] = field(default_factory=list)  # (host, method, path?query)
    internal_hits: list[str] = field(default_factory=list)

    def url(self, path: str, host: str = "shop.test") -> str:
        return f"http://{host}:{self.port}{path}"

    def page(self, path: str, body: str, *, host: str = "shop.test", title: str = "Shop") -> str:
        self.pages[(host, path)] = (
            f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>'
            f"<body>{body}</body></html>"
        )
        return self.url(path, host)

    def hidden_redirect(self, target: str) -> str:
        """A 302 whose target is not visible in its own URL."""
        key = str(len(self.redirects))
        self.redirects[key] = target
        return self.url(f"/go/{key}")

    def attacker_hits(self) -> list[tuple[str, str, str]]:
        return [h for h in self.hits if h[0] == "attacker.test"]

    def clear(self) -> None:
        self.hits.clear()
        self.internal_hits.clear()

    async def resolve(self, host: str, port: int) -> list[IPAddress]:
        if host in PUBLIC_HOSTS or host == "internal.test":
            return [LOOPBACK]
        if host == "metadata.test":
            return [ipaddress.ip_address("169.254.169.254")]
        raise OSError("NXDOMAIN")

    def block(self, ip: IPAddress, port: int) -> str | None:
        # The stand-in public web lives on loopback; everything else gets the real policy.
        if ip == LOOPBACK and port == self.port:
            return None
        return default_block(ip, port)


def _handler(web: Web) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def _serve(self, method: str) -> None:
            host = self.headers.get("Host", "").split(":")[0]
            url = urlsplit(self.path)
            web.hits.append((host, method, self.path))
            if url.path.startswith("/go/"):
                self.send_response(302)
                self.send_header("Location", web.redirects[url.path.removeprefix("/go/")])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if url.path == "/to":  # the classic open redirect
                self.send_response(302)
                self.send_header("Location", parse_qs(url.query)["url"][0])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = web.pages.get((host, url.path))
            if body is None:
                body = "<!doctype html><title>ok</title><p>nothing here</p>"
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._serve("GET")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self._serve("POST")

    return Handler


@contextmanager
def serve_web() -> Iterator[Web]:
    web = Web()

    class Internal(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            web.internal_hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "6")
            self.end_headers()
            self.wfile.write(b"secret")

    site = ThreadingHTTPServer(("127.0.0.1", 0), _handler(web))
    internal = ThreadingHTTPServer(("127.0.0.1", 0), Internal)
    web.port, web.internal_port = site.server_address[1], internal.server_address[1]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (site, internal)]
    for t in threads:
        t.start()
    try:
        yield web
    finally:
        for s in (site, internal):
            s.shutdown()
            s.server_close()


_CMD = re.compile(r"AGENT-(CMD|LOOP)\[(\w+)\|([^|\]]*)(?:\|([^\]]*))?\]")


def element_index(prompt: str, name: str) -> int | None:
    """Index of the element shown as ``[N]<tag …>name`` in the observation."""
    m = re.search(rf"\[(\d+)\](?:\(iframe [^)]*\))?<[^>]*>{re.escape(name)}", prompt)
    return int(m.group(1)) if m else None


class ObedientModel(BaseChatModel):
    """Executes the page's instructions. ``done`` is the answer it gives when none are left."""

    done: str = "finished"
    prompts: list[str] = Field(default_factory=list)  # every decide prompt, for assertions
    executed: list[str] = Field(default_factory=list)
    extract_prompts: list[str] = Field(default_factory=list)
    extract_answer: str = "NOT FOUND"

    @property
    def _llm_type(self) -> str:
        return "obedient"

    def _respond(self, tools: set[str], messages: Sequence[BaseMessage]) -> AIMessage:
        if "Plan" in tools:
            return PLAN
        if "Reflection" in tools:
            return tool("Reflection", diagnosis="stuck", revised_plan=["keep going"], advice="")
        if not tools:  # an Extract question
            self.extract_prompts.append(str(messages[-1].content))
            return text(self.extract_answer)
        prompt = str(messages[-1].content)
        self.prompts.append(prompt)
        for m in _CMD.finditer(prompt):
            kind, name, target, value = m.group(1), m.group(2), m.group(3), m.group(4)
            if kind == "CMD" and m.group(0) in self.executed:
                continue
            if call := self._call(name, target, value, prompt):
                self.executed.append(m.group(0))
                return call
        return tool("Done", result=self.done)

    @staticmethod
    def _call(name: str, target: str, value: str | None, prompt: str) -> AIMessage | None:
        if name == "Navigate":
            return tool("Navigate", url=target)
        if name == "Extract":
            return tool("Extract", query=target)
        index = element_index(prompt, target)
        if index is None:
            return None  # the element isn't in what the model sees: nothing to obey
        if name == "Click":
            return tool("Click", index=index)
        if name in ("Type", "TypeSubmit"):
            return tool("TypeText", index=index, text=value or "", submit=name == "TypeSubmit")
        return None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(set(), messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop)

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        names = {t["function"]["name"] for t in tools}
        respond: Callable[[Any], AIMessage] = lambda messages: self._respond(names, messages)  # noqa: E731

        async def arespond(messages: Any) -> AIMessage:
            return respond(messages)

        return RunnableLambda(respond, afunc=arespond)
