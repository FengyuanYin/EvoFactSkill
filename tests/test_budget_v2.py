from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from evofact.core.budget_models import (
    BudgetLimits,
    BudgetRequest,
    CostStatus,
    UsageDetails,
)
from evofact.governance.pricing_policy import ModelPrice, PricingTable, load_pricing_table
from evofact.runtime.budget import BudgetExceeded, BudgetManager
from evofact.runtime.pricing import parse_openai_usage, price_usage


def test_budget_reservation_reconciles_decimal_usage() -> None:
    async def scenario():
        manager = BudgetManager(BudgetLimits(max_calls_per_sample=2, max_calls_per_run=2))
        reservation = await manager.reserve(
            "sample", BudgetRequest(tokens=20, cost=Decimal("0.02"))
        )
        await manager.reconcile(
            reservation,
            UsageDetails(
                calls=1,
                input_tokens=10,
                output_tokens=5,
                cost=Decimal("0.015"),
                cost_status=CostStatus.ACTUAL,
            ),
        )
        return manager.snapshot("sample")

    snapshot = asyncio.run(scenario())
    assert snapshot.calls_used == 1
    assert snapshot.tokens_used == 15
    assert snapshot.cost_used == Decimal("0.015")
    assert snapshot.active_reservations == 0


def test_optional_call_cannot_consume_reserved_judge_call() -> None:
    async def scenario():
        limits = BudgetLimits(max_calls_per_sample=1, judge_reserved_calls=1)
        manager = BudgetManager(limits)
        with pytest.raises(BudgetExceeded):
            await manager.reserve("sample", BudgetRequest(required=False))

    asyncio.run(scenario())


def test_openai_usage_and_pricing_include_cache_and_reasoning() -> None:
    usage = parse_openai_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 5},
            }
        },
        provider="provider",
        model="model",
        latency_ms=12,
    )
    priced = price_usage(
        usage,
        PricingTable(
            "provider",
            {
                "model": ModelPrice(
                    Decimal("1"),
                    Decimal("2"),
                    Decimal("0.1"),
                    Decimal("3"),
                )
            },
        ),
    )
    # Provider totals include cached input and reasoning output; subsets must not be billed twice.
    assert priced.cost == Decimal("0.000109")
    assert priced.cost_status == CostStatus.ACTUAL


def test_missing_usage_is_not_silently_actual_zero() -> None:
    usage = parse_openai_usage({}, provider="provider", model="model", latency_ms=1)
    assert usage.cost is None
    assert usage.cost_status == CostStatus.UNAVAILABLE


def test_budget_checkpoint_restore_preserves_run_and_sample_usage() -> None:
    async def scenario():
        manager = BudgetManager(BudgetLimits(max_calls_per_run=2, max_calls_per_sample=2))
        reservation = await manager.reserve("sample", BudgetRequest(calls=1, tokens=10))
        await manager.reconcile(
            reservation,
            UsageDetails(calls=1, input_tokens=7, output_tokens=2),
        )
        restored = BudgetManager(manager.limits)
        restored.restore(manager.export_state())
        return restored.snapshot(), restored.snapshot("sample")

    run, sample = asyncio.run(scenario())
    assert run.calls_used == sample.calls_used == 1
    assert run.tokens_used == sample.tokens_used == 9
    assert run.unavailable_cost_calls == sample.unavailable_cost_calls == 1


def test_pricing_loader_rejects_duplicate_model_keys(tmp_path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(
        '{"version":"pricing_v1","provider":"p","models":{'
        '"m":{"input_per_million":1,"output_per_million":2},'
        '"m":{"input_per_million":3,"output_per_million":4}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate pricing key: m"):
        load_pricing_table(path)
