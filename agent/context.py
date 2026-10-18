"""Run-scoped dependencies, passed through LangGraph's runtime context (never checkpointed)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from browser.service import BrowserService
from core.config import Settings
from core.costs import PriceTable
from core.secrets import SecretResolver


@dataclass(frozen=True)
class AgentContext:
    browser: BrowserService
    llm_decide: BaseChatModel
    llm_small: BaseChatModel
    settings: Settings
    prices: PriceTable
    clock: Callable[[], float] = field(default=time.time)
    # Looks up {{SECRET:name}} values; None = runs cannot use secrets.
    secrets: SecretResolver | None = None
