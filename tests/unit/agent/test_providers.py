"""OpenAI, Anthropic and OpenRouter: model defaults, keys, request payloads and prices — offline."""

from pathlib import Path

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.llm import forced_tool_choice, make_chat_model
from agent.tools import ACTION_TOOL_SPECS
from core.costs import PriceTable, UnknownModelPriceError
from tests.conftest import make_settings

ROOT = Path("unused")
MESSAGES = [
    SystemMessage("sys"),
    HumanMessage(
        content=[
            {"type": "text", "text": "page"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
        ]
    ),
]
NAMES = ["Navigate", "Click", "TypeText", "Scroll", "Extract", "Done", "AskHuman"]


def test_models_default_per_provider_unless_set() -> None:
    assert make_settings(ROOT).model_decide == "gpt-4.1"
    a = make_settings(ROOT, llm_provider="anthropic")
    assert (a.model_decide, a.model_small) == ("claude-sonnet-4-5", "claude-haiku-4-5")
    o = make_settings(ROOT, llm_provider="openrouter", model_small="anthropic/claude-haiku-4.5")
    assert (o.model_decide, o.model_small) == ("openai/gpt-4.1", "anthropic/claude-haiku-4.5")
    keep = make_settings(ROOT, llm_provider="anthropic", model_decide="claude-opus-4-1")
    assert keep.model_decide == "claude-opus-4-1" and keep.model_small == "claude-haiku-4-5"


def test_blank_model_names_mean_the_default() -> None:
    s = make_settings(ROOT, llm_provider="anthropic", model_decide="", model_small=" ")
    assert (s.model_decide, s.model_small) == ("claude-sonnet-4-5", "claude-haiku-4-5")


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "gpt-4.1"),
        ("anthropic", "openai/gpt-4.1"),
        ("openai", "claude-sonnet-4-5"),
        ("openrouter", "gpt-4.1"),
    ],
)
def test_model_names_from_another_provider_are_rejected(provider: str, model: str) -> None:
    with pytest.raises(ValueError, match=r"BU_MODEL_DECIDE=.*unset it"):
        make_settings(ROOT, llm_provider=provider, model_decide=model)


def test_each_provider_needs_its_own_key() -> None:
    s = make_settings(ROOT, llm_provider="anthropic", openai_api_key="sk-openai-only")
    assert s.llm_api_key is None and s.llm_key_name == "BU_ANTHROPIC_API_KEY"
    with pytest.raises(RuntimeError, match="BU_ANTHROPIC_API_KEY is not set"):
        make_chat_model(s, s.model_decide)
    blank = make_settings(ROOT, llm_provider="openrouter", openrouter_api_key="  ")
    assert blank.llm_api_key is None  # a blank CI secret counts as unset
    ok = make_settings(ROOT, llm_provider="openrouter", openrouter_api_key="sk-or-test")
    assert ok.llm_api_key is not None and ok.llm_key_name == "BU_OPENROUTER_API_KEY"


def test_anthropic_payload_forces_a_tool_without_parallel_calls() -> None:
    s = make_settings(ROOT, llm_provider="anthropic", anthropic_api_key="sk-ant-test")
    llm = make_chat_model(s, s.model_decide)
    assert isinstance(llm, ChatAnthropic) and forced_tool_choice(llm) == "any"
    bound = llm.bind_tools(ACTION_TOOL_SPECS, tool_choice="any", parallel_tool_calls=False)
    payload = llm._get_request_payload(MESSAGES, **bound.kwargs)  # type: ignore[attr-defined]
    assert payload["model"] == "claude-sonnet-4-5"
    assert payload["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    assert [t["name"] for t in payload["tools"]] == NAMES
    assert payload["max_tokens"] == 4096
    assert payload["messages"][0]["content"][1]["type"] == "image"  # screenshots still work


def test_openrouter_is_openai_compatible_at_its_own_url() -> None:
    s = make_settings(ROOT, llm_provider="openrouter", openrouter_api_key="sk-or-test")
    llm = make_chat_model(s, "anthropic/claude-sonnet-4.5")
    assert isinstance(llm, ChatOpenAI) and forced_tool_choice(llm) == "required"
    assert str(llm.openai_api_base) == "https://openrouter.ai/api/v1"
    bound = llm.bind_tools(ACTION_TOOL_SPECS, tool_choice="required", parallel_tool_calls=False)
    payload = llm._get_request_payload(MESSAGES, **bound.kwargs)  # type: ignore[attr-defined]
    assert payload["model"] == "anthropic/claude-sonnet-4.5"
    assert payload["tool_choice"] == "required" and payload["parallel_tool_calls"] is False


def test_openai_still_uses_the_default_endpoint() -> None:
    s = make_settings(ROOT, openai_api_key="sk-test")
    llm = make_chat_model(s, "gpt-4.1")
    assert isinstance(llm, ChatOpenAI) and "openrouter" not in str(llm.openai_api_base or "")


@pytest.mark.parametrize(
    ("model", "price"),
    [
        ("claude-sonnet-4-5", (3.0, 15.0)),
        ("claude-sonnet-4-5-20250929", (3.0, 15.0)),  # dated snapshot
        ("claude-haiku-4-5", (1.0, 5.0)),
        ("anthropic/claude-sonnet-4.5", (3.0, 15.0)),  # OpenRouter spelling
        ("openai/gpt-4.1", (2.0, 8.0)),
        ("openai/gpt-4.1-mini", (0.4, 1.6)),
        ("openai/gpt-4.1:extended", (2.0, 8.0)),
    ],
)
def test_prices_resolve_across_providers(model: str, price: tuple[float, float]) -> None:
    assert PriceTable.from_settings(make_settings(ROOT)).price(model) == price


def test_unknown_models_need_an_explicit_price() -> None:
    table = PriceTable.from_settings(make_settings(ROOT))
    with pytest.raises(UnknownModelPriceError, match="BU_MODEL_PRICES"):
        table.price("meta-llama/llama-4-maverick")
    custom = PriceTable.from_settings(
        make_settings(ROOT, model_prices={"llama-4-maverick": (0.2, 0.6)})
    )
    assert custom.price("meta-llama/llama-4-maverick") == (0.2, 0.6)
