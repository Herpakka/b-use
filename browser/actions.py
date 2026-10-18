"""The only things a model can ask the browser to do.

Each class doubles as a tool schema (class name = tool name, docstring = description), so keep
docstrings short and imperative. There is deliberately no eval / run-JavaScript action.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class BaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Navigate(BaseAction):
    """Open an absolute http(s) URL in the current tab. Only the run's allowed domains load."""

    kind: Literal["navigate"] = "navigate"
    url: str = Field(description="Absolute http(s) URL.")


class Click(BaseAction):
    """Click the element shown as [index] in the current page observation."""

    kind: Literal["click"] = "click"
    index: int = Field(ge=1, description="Element number shown as [N] in the observation.")


class TypeText(BaseAction):
    """Replace the content of the input/textarea at [index] with text.

    For a <select>, text is the label of the option to choose.
    """

    kind: Literal["type_text"] = "type_text"
    index: int = Field(ge=1, description="Element number shown as [N] in the observation.")
    text: str
    submit: bool = Field(default=False, description="Press Enter after typing.")


class Scroll(BaseAction):
    """Scroll the page to reveal more content."""

    kind: Literal["scroll"] = "scroll"
    direction: Literal["down", "up"] = "down"
    pages: float = Field(default=1.0, gt=0, le=5, description="How many viewport heights.")


class Extract(BaseAction):
    """Read the full text of the current page to answer a question about its content."""

    kind: Literal["extract"] = "extract"
    query: str = Field(description="What to find on the page.")


class Done(BaseAction):
    """Finish the task. Put the complete final answer in result."""

    kind: Literal["done"] = "done"
    result: str
    success: bool = Field(default=True, description="False if the task could not be completed.")


AnyAction = Navigate | Click | TypeText | Scroll | Extract | Done
ACTION_TYPES: tuple[type[BaseAction], ...] = (Navigate, Click, TypeText, Scroll, Extract, Done)

ActionAdapter: TypeAdapter[AnyAction] = TypeAdapter(
    Annotated[AnyAction, Field(discriminator="kind")]
)
