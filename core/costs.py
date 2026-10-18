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
    # Anthropic. Newer models are not listed: set theirs in BU_MODEL_PRICES.
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-1": (15.00, 75.00),
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
        # OpenRouter names are "vendor/model" ("anthropic/claude-sonnet-4.5", "openai/gpt-4.1");
        # the vendor's own price table applies, spelling versions with dots or dashes alike.
        base = model.split("/", 1)[-1].split(":", 1)[0]
        for name in (model, base, base.replace(".", "-")):
            if name in self.prices:
                return self.prices[name]
        # Dated snapshots ("gpt-4.1-mini-2025-04-14") use their family's price.
        matches = [
            k
            for name in (base, base.replace(".", "-"))
            for k in self.prices
            if name.startswith(k + "-")
        ]
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
