from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from evofact.governance import PRICING_SCHEMA_VERSION


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal
    cache_per_million: Decimal = Decimal("0")
    reasoning_per_million: Decimal | None = None


@dataclass(frozen=True)
class PricingTable:
    provider: str
    models: dict[str, ModelPrice]
    version: str = PRICING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != PRICING_SCHEMA_VERSION:
            raise ValueError(f"unsupported pricing schema: {self.version}")
