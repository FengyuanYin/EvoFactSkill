from __future__ import annotations

from decimal import Decimal

from evofact.core.budget_models import CostStatus, UsageDetails
from evofact.governance.pricing_policy import PricingTable

MILLION = Decimal(1_000_000)


def price_usage(usage: UsageDetails, table: PricingTable) -> UsageDetails:
    price = table.models.get(usage.model or "")
    if price is None:
        return UsageDetails(
            **{
                **usage.model_dump(),
                "cost": None,
                "cost_status": CostStatus.UNAVAILABLE,
                "pricing_version": table.version,
            }
        )
    reasoning_price = price.reasoning_per_million or price.output_per_million
    uncached_input_tokens = max(usage.input_tokens - usage.cached_tokens, 0)
    visible_output_tokens = max(usage.output_tokens - usage.reasoning_tokens, 0)
    cost = (
        Decimal(uncached_input_tokens) * price.input_per_million
        + Decimal(visible_output_tokens) * price.output_per_million
        + Decimal(usage.cached_tokens) * price.cache_per_million
        + Decimal(usage.reasoning_tokens) * reasoning_price
    ) / MILLION
    return UsageDetails(
        **{
            **usage.model_dump(),
            "cost": cost,
            "cost_status": usage.cost_status
            if usage.cost_status != CostStatus.UNAVAILABLE
            else CostStatus.ACTUAL,
            "pricing_version": table.version,
        }
    )


def parse_openai_usage(
    data: dict, *, provider: str, model: str, latency_ms: float, retries: int = 0
) -> UsageDetails:
    usage = data.get("usage")
    token_fields = {
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
    }
    if not isinstance(usage, dict) or not token_fields.intersection(usage):
        return UsageDetails(
            calls=1,
            retries=retries,
            latency_ms=latency_ms,
            cost_status=CostStatus.UNAVAILABLE,
            provider=provider,
            model=model,
        )
    input_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    output_details = (
        usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    )
    return UsageDetails(
        calls=1,
        retries=retries,
        input_tokens=int(usage.get("prompt_tokens", usage.get("input_tokens", 0))),
        output_tokens=int(usage.get("completion_tokens", usage.get("output_tokens", 0))),
        cached_tokens=int(input_details.get("cached_tokens", 0)),
        reasoning_tokens=int(output_details.get("reasoning_tokens", 0)),
        latency_ms=latency_ms,
        cost_status=CostStatus.ACTUAL,
        provider=provider,
        model=model,
    )
