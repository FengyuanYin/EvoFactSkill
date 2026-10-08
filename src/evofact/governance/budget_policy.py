from __future__ import annotations

from dataclasses import dataclass

from evofact.core.budget_models import BudgetLimits, BudgetRequest, BudgetSnapshot


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str = ""


def evaluate_budget(
    request: BudgetRequest,
    sample: BudgetSnapshot,
    run: BudgetSnapshot,
    limits: BudgetLimits,
    *,
    judge: bool = False,
) -> BudgetDecision:
    reserved_calls = 0 if judge or request.required else limits.judge_reserved_calls
    reserved_tokens = 0 if judge or request.required else limits.judge_reserved_tokens
    reserved_cost = 0 if judge or request.required else limits.judge_reserved_cost
    if (
        sample.calls_used + sample.calls_reserved + request.calls + reserved_calls
        > limits.max_calls_per_sample
    ):
        return BudgetDecision(False, "sample call budget exhausted")
    if limits.max_tokens_per_sample is not None and (
        sample.tokens_used + sample.tokens_reserved + request.tokens + reserved_tokens
        > limits.max_tokens_per_sample
    ):
        return BudgetDecision(False, "sample token budget exhausted")
    if (
        sample.cost_used + sample.cost_reserved + request.cost + reserved_cost
        > limits.max_cost_per_sample
    ):
        return BudgetDecision(False, "sample cost budget exhausted")
    if run.calls_used + run.calls_reserved + request.calls > limits.max_calls_per_run:
        return BudgetDecision(False, "run call budget exhausted")
    if (
        limits.max_tokens_per_run is not None
        and run.tokens_used + run.tokens_reserved + request.tokens > limits.max_tokens_per_run
    ):
        return BudgetDecision(False, "run token budget exhausted")
    if run.cost_used + run.cost_reserved + request.cost > limits.max_cost_per_run:
        return BudgetDecision(False, "run cost budget exhausted")
    return BudgetDecision(True)
