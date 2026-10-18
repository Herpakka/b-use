"""Real LLM + real Chromium on one public task. Needs the key of BU_LLM_PROVIDER; ~1 cent."""

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
    if settings.llm_api_key is None:
        pytest.skip(f"{settings.llm_key_name} not set")
    suite = Suite.model_validate(yaml.safe_load(SUITE.read_text(encoding="utf-8")))
    tasks = [t for t in suite.tasks if t.id == "books-price"]
    [result] = await run_suite(tasks, settings, memory=True, verbose=True)
    assert result.passed, result.why
    assert result.steps <= 6
    assert result.cost_usd < settings.budget_usd
