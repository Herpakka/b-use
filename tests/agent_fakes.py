"""Deterministic stand-ins for the LLM and the browser so the graph can be tested offline."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from agent.context import AgentContext
from agent.graph import AgentGraph, compile_graph
from agent.runner import HumanHandler, drive, initial_state, limits_from
from browser.actions import AnyAction, Extract, Navigate
from browser.models import (
    ActionResult,
    BrowserHealth,
    ElementInfo,
    Observation,
    ScrollState,
    SessionInfo,
)
from browser.service import SessionNotFoundError
from core.costs import PriceTable
from core.secrets import Secret, SecretResolver
from tests.conftest import make_settings

JPEG_MAGIC = bytes([0xFF, 0xD8, 0xFF])
USAGE = UsageMetadata(input_tokens=1000, output_tokens=50, total_tokens=1050)
_ids = itertools.count()


def tool(name: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}],
        usage_metadata=USAGE,
    )


def text(content: str) -> AIMessage:
    return AIMessage(content=content, usage_metadata=USAGE)


PLAN = tool("Plan", subgoals=["open the page", "find the answer"])


class ScriptedChatModel(BaseChatModel):
    """Replies with the next scripted message; records every prompt it was given."""

    script: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    delay: float = 0.0  # seconds each call takes (to test cancel/recovery mid-run)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError(f"LLM script exhausted after {len(self.calls) - 1} calls")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._generate(messages, stop, **kwargs)

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


def page(
    url: str, *elements: tuple[str, str] | tuple[str, str, str], body: str = ""
) -> Observation:
    """Observation with elements given as (tag, name[, type]); indices start at 1."""
    infos = [
        ElementInfo(index=i, tag=e[0], name=e[1], type=e[2] if len(e) > 2 else None)
        for i, e in enumerate(elements, 1)
    ]
    outline: list[int | str] = [body] if body else []
    outline += [e.index for e in infos]
    return Observation(
        dom_version="v0",
        url=url,
        title=url,
        elements=infos,
        outline=outline,
        scroll=ScrollState(viewport_height=800),
    )


class FakeBrowser:
    def __init__(
        self,
        pages: dict[str, Observation] | None = None,
        *,
        results: list[ActionResult] | None = None,
        page_text: str = "",
    ) -> None:
        self.pages = pages or {}
        self.results = list(results or [])
        self.page_text = page_text
        self.url = "about:blank"
        self.sessions: set[str] = set()
        self.executed: list[AnyAction] = []
        self.closed: list[str] = []  # closed with cookies saved
        self.forgotten: list[str] = []  # closed with cookies deleted
        self.held: set[str] = set()
        self.secrets: list[dict[str, Secret]] = []  # the values handed over with each action
        self._version = 0

    def _require(self, session_id: str) -> None:
        if session_id not in self.sessions:
            raise SessionNotFoundError(session_id)

    async def open(
        self, session_id: str, *, allowed_domains: Sequence[str], vision: bool = False
    ) -> None:
        self.sessions.add(session_id)

    async def has_session(self, session_id: str) -> bool:
        return session_id in self.sessions

    async def observe(self, session_id: str, *, screenshot: bool = False) -> Observation:
        self._require(session_id)
        self._version += 1
        base = self.pages.get(self.url) or page(self.url)
        return base.model_copy(update={"dom_version": f"v{self._version}", "url": self.url})

    async def execute(
        self,
        session_id: str,
        action: AnyAction,
        *,
        expected_version: str | None = None,
        secrets: Mapping[str, Secret] | None = None,
    ) -> ActionResult:
        self._require(session_id)
        self.executed.append(action)
        self.secrets.append(dict(secrets or {}))
        if isinstance(action, Navigate):
            self.url = action.url
            return ActionResult(ok=True, message=f"navigate {action.url} → ok", url=self.url)
        if self.results:
            result = self.results.pop(0)
            if result.url:
                self.url = result.url
            return result.model_copy(update={"url": self.url})
        if isinstance(action, Extract):
            return ActionResult(
                ok=True,
                message=f'extract "{action.query}" → read',
                extracted_text=self.page_text,
                url=self.url,
            )
        return ActionResult(ok=True, message=f"{action.kind} → ok", url=self.url)

    async def screenshot(self, session_id: str) -> bytes:
        self._require(session_id)
        return b"\xff\xd8\xff"

    async def close(self, session_id: str) -> None:
        self.sessions.discard(session_id)
        self.closed.append(session_id)

    async def forget(self, session_id: str) -> None:
        self.sessions.discard(session_id)
        self.forgotten.append(session_id)

    async def hold(self, session_id: str, held: bool) -> None:
        (self.held.add if held else self.held.discard)(session_id)

    async def list_sessions(self) -> list[SessionInfo]:
        return [
            SessionInfo(id=sid, url=self.url, tabs=1, idle_s=0.0, busy=False)
            for sid in sorted(self.sessions)
        ]

    async def health(self) -> BrowserHealth:
        return BrowserHealth(ok=True, detail="ok", sessions=len(self.sessions), max_sessions=4)

    async def screencast(self, session_id: str) -> AsyncIterator[bytes]:
        self._require(session_id)
        for i in range(3):
            yield JPEG_MAGIC + bytes([i])


class StepClock:
    """Each call returns a time ``step`` seconds after the previous one."""

    def __init__(self, step: float = 0.0, start: float = 1000.0) -> None:
        self.now, self.step = start, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def make_ctx(
    browser: FakeBrowser,
    decide: list[AIMessage],
    small: list[AIMessage] | None = None,
    *,
    clock: Callable[[], float] | None = None,
    secrets: SecretResolver | None = None,
    **settings: object,
) -> AgentContext:
    s = make_settings(Path("unused-sessions"), **settings)
    return AgentContext(
        browser=browser,
        llm_decide=ScriptedChatModel(script=list(decide)),
        llm_small=ScriptedChatModel(script=list(small or [])),
        settings=s,
        prices=PriceTable.from_settings(s),
        clock=clock or StepClock(),
        secrets=secrets,
    )


async def run_task(
    ctx: AgentContext,
    *,
    task: str = "find the answer",
    start_url: str | None = "https://shop.test/",
    on_human: HumanHandler | None = None,
    saver: BaseCheckpointSaver[Any] | None = None,
    run_id: str | None = None,
    secret_scope: str = "",
) -> tuple[dict[str, Any], dict[str, Any] | None, AgentGraph]:
    graph = compile_graph(saver or InMemorySaver())
    limits = limits_from(ctx.settings)
    state = initial_state(
        ctx,
        task=task,
        allowed_domains=["shop.test"],
        start_url=start_url,
        run_id=run_id,
        limits=limits,
        secret_scope=secret_scope,
    )
    final, pending = await drive(
        graph, ctx, state["run_id"], state, max_steps=limits["max_steps"], on_human=on_human
    )
    return final, pending, graph


def decide_prompts(ctx: AgentContext) -> list[str]:
    """The human-message text of every LLM call made with the decide model."""
    llm = ctx.llm_decide
    assert isinstance(llm, ScriptedChatModel)
    return [str(call[-1].content) for call in llm.calls]
