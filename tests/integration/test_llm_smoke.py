"""Real OpenAI + real Chromium on one public task. Needs BU_OPENAI_API_KEY; costs ~1 cent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config import get_settings
from evals.runner import Suite, run_suite

pytestmark = [pytest.mark.llm, pytest.mark.live, pytest.mark.browser]

SUITE = Path(__file__).parents[2] / "evals" / "smoke.yaml"


async def test_books_price_end_to_end() -> None:
    settings = get_settings()
    if settings.openai_api_key is None:
        pytest.skip("BU_OPENAI_API_KEY not set")
    suite = Suite.model_validate(yaml.safe_load(SUITE.read_text(encoding="utf-8")))
    tasks = [t for t in suite.tasks if t.id == "books-price"]
    [result] = await run_suite(tasks, settings, memory=True, verbose=True)
    assert result.passed, result.why
    assert result.steps <= 6
    assert result.cost_usd < settings.budget_usd
