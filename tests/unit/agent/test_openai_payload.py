"""Build real OpenAI request payloads offline: catches schema/binding mistakes without a key."""

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.llm import make_chat_model
from agent.tools import ACTION_TOOL_SPECS
from tests.conftest import make_settings


def test_decide_request_payload() -> None:
    settings = make_settings(Path("unused"), openai_api_key="sk-test-not-a-real-key")
    llm = make_chat_model(settings, "gpt-4.1")
    assert isinstance(llm, ChatOpenAI)
    bound = llm.bind_tools(ACTION_TOOL_SPECS, tool_choice="required", parallel_tool_calls=False)
    kwargs = bound.kwargs  # type: ignore[attr-defined]
    messages = [
        SystemMessage("sys"),
        HumanMessage(
            content=[
                {"type": "text", "text": "page"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            ]
        ),
    ]
    payload = llm._get_request_payload(messages, **kwargs)

    assert payload["model"] == "gpt-4.1"
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names == ["Navigate", "Click", "TypeText", "Scroll", "Extract", "Done", "AskHuman"]
    user = payload["messages"][1]
    assert user["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
