from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from evofact.governance import PRICING_SCHEMA_VERSION


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal
    cache_per_million: Decimal = Decimal("0")
    reasoning_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        values = (
            self.input_per_million,
            self.output_per_million,
            self.cache_per_million,
        )
        if any(value < 0 for value in values):
            raise ValueError("model prices must not be negative")
        if self.reasoning_per_million is not None and self.reasoning_per_million < 0:
            raise ValueError("reasoning price must not be negative")


@dataclass(frozen=True)
class PricingTable:
    provider: str
    models: dict[str, ModelPrice]
    version: str = PRICING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != PRICING_SCHEMA_VERSION:
            raise ValueError(f"unsupported pricing schema: {self.version}")
        if not self.provider.strip() or not self.models:
            raise ValueError("pricing table requires a provider and at least one model")

    @property
    def identity(self) -> str:
        payload = {
            "version": self.version,
            "provider": self.provider,
            "models": {
                name: {
                    "input_per_million": str(price.input_per_million),
                    "output_per_million": str(price.output_per_million),
                    "cache_per_million": str(price.cache_per_million),
                    "reasoning_per_million": (
                        str(price.reasoning_per_million)
                        if price.reasoning_per_million is not None
                        else None
                    ),
                }
                for name, price in sorted(self.models.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_pricing_table(path: str | Path) -> PricingTable:
    """Load a strict JSON pricing table without inventing missing prices."""

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate pricing key: {key}")
            value[key] = item
        return value

    raw = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(raw, dict) or set(raw) != {"version", "provider", "models"}:
        raise ValueError("pricing table must contain exactly version, provider, and models")
    if not isinstance(raw["models"], dict) or not raw["models"]:
        raise ValueError("pricing table models must be a non-empty object")
    allowed = {
        "input_per_million",
        "output_per_million",
        "cache_per_million",
        "reasoning_per_million",
    }
    models = {}
    for name, value in raw["models"].items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, dict):
            raise ValueError("invalid pricing model entry")
        if set(value) - allowed or not {
            "input_per_million",
            "output_per_million",
        } <= set(value):
            raise ValueError(f"invalid pricing fields for model {name}")
        converted = {
            key: Decimal(str(item)) if item is not None else None for key, item in value.items()
        }
        models[name] = ModelPrice(**converted)
    return PricingTable(str(raw["provider"]), models, str(raw["version"]))
