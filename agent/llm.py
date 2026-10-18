"""Thin helpers over LangChain chat models: forced single tool calls and usage accounting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from agent.context import AgentContext
from core.config import Settings


class NoToolCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    tokens_in: int
    tokens_out: int


def make_chat_model(settings: Settings, model: str) -> BaseChatModel:
    if settings.openai_api_key is None:
        raise RuntimeError("BU_OPENAI_API_KEY (or OPENAI_API_KEY) is not set")
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )


def _usage(message: BaseMessage) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
    )


async def call_tool(
    llm: BaseChatModel, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
) -> ToolCall:
    """Invoke ``llm`` forcing exactly one tool call; return it with its token usage."""
    bound = llm.bind_tools(list(tools), tool_choice="required", parallel_tool_calls=False)
    message = await bound.ainvoke(list(messages))
    tokens_in, tokens_out = _usage(message)
    calls = message.tool_calls if isinstance(message, AIMessage) else []
    if not calls:
        raise NoToolCallError(message_text(message)[:200] or "model returned no tool call")
    first = calls[0]
    return ToolCall(
        name=first["name"], args=dict(first["args"]), tokens_in=tokens_in, tokens_out=tokens_out
    )


async def ask_text(llm: BaseChatModel, messages: Sequence[BaseMessage]) -> tuple[str, int, int]:
    message = await llm.ainvoke(list(messages))
    tokens_in, tokens_out = _usage(message)
    return message_text(message).strip(), tokens_in, tokens_out


def usage_update(ctx: AgentContext, model: str, tokens_in: int, tokens_out: int) -> dict[str, Any]:
    """State delta for the additive cost/token channels."""
    return {
        "cost_usd": ctx.prices.cost(model, tokens_in, tokens_out),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
