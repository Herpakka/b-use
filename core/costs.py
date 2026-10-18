"""Token prices and cost accounting. USD per 1M tokens as (input, output)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.config import Settings

# Override or extend with BU_MODEL_PRICES; check https://openai.com/api/pricing before relying
# on these for budgeting.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


class UnknownModelPriceError(KeyError):
    pass


@dataclass(frozen=True)
class PriceTable:
    prices: Mapping[str, tuple[float, float]]

    @classmethod
    def from_settings(cls, settings: Settings) -> PriceTable:
        return cls({**DEFAULT_PRICES, **settings.model_prices})

    def price(self, model: str) -> tuple[float, float]:
        if model in self.prices:
            return self.prices[model]
        # Dated snapshots ("gpt-4.1-mini-2025-04-14") use their family's price.
        matches = [k for k in self.prices if model.startswith(k + "-")]
        if not matches:
            raise UnknownModelPriceError(
                f"no price for model {model!r}; set it in BU_MODEL_PRICES so the budget works"
            )
        return self.prices[max(matches, key=len)]

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        p_in, p_out = self.price(model)
        return (tokens_in * p_in + tokens_out * p_out) / 1_000_000

    def require(self, *models: str) -> None:
        """Fail fast at startup instead of running with an uncounted budget."""
        for m in models:
            self.price(m)
