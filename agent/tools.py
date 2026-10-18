"""Tool schemas the model sees: the browser actions plus agent-level tools."""

from __future__ import annotations

from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, ValidationError

from browser.actions import ACTION_TYPES, AnyAction


class AskHuman(BaseModel):
    """Ask the human operator: a login, captcha or 2FA, missing information, or a decision
    only they can make. Never guess credentials."""

    question: str


class Plan(BaseModel):
    """Record the plan for the task."""

    subgoals: list[str] = Field(min_length=1, max_length=8, description="2-6 short subgoals.")


class Reflection(BaseModel):
    """Diagnose why progress stalled and revise the plan."""

    diagnosis: str = Field(description="What went wrong, in one or two sentences.")
    revised_plan: list[str] = Field(description="Remaining subgoals, revised.")
    advice: str = Field(description="Concrete advice for the very next step.")


AgentTool = AnyAction | AskHuman
_TOOL_CLASSES: tuple[type[BaseModel], ...] = (*ACTION_TYPES, AskHuman)
AGENT_TOOLS: dict[str, type[BaseModel]] = {cls.__name__: cls for cls in _TOOL_CLASSES}


class ToolParseError(ValueError):
    pass


def tool_spec(cls: type[BaseModel]) -> dict[str, Any]:
    """OpenAI function spec for ``cls`` minus the internal ``kind`` discriminator."""
    spec = convert_to_openai_tool(cls)
    params = spec["function"]["parameters"]
    params.get("properties", {}).pop("kind", None)
    if "required" in params:
        params["required"] = [r for r in params["required"] if r != "kind"]
    return spec


ACTION_TOOL_SPECS: list[dict[str, Any]] = [tool_spec(c) for c in AGENT_TOOLS.values()]
PLAN_TOOL_SPEC = tool_spec(Plan)
REFLECT_TOOL_SPEC = tool_spec(Reflection)


def parse_agent_tool(name: str, args: dict[str, Any]) -> AgentTool:
    cls = AGENT_TOOLS.get(name)
    if cls is None:
        raise ToolParseError(f"unknown tool {name!r}")
    try:
        parsed = cls.model_validate(args)
    except ValidationError as err:
        first = err.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or name
        raise ToolParseError(f"{loc}: {first['msg']}") from err
    assert isinstance(parsed, AnyAction | AskHuman)
    return parsed
