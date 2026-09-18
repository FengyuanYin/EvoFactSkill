from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .models import ModelMixin


class CostStatus(StrEnum):
    ACTUAL = "actual"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BudgetLimits(ModelMixin):
    max_nodes: int = 8
    max_depth: int = 4
    max_calls_per_sample: int = 8
    max_tokens_per_sample: int = 12000
    max_cost_per_sample: Decimal = Decimal("1")
    max_calls_per_run: int = 1000
    max_tokens_per_run: int = 1_000_000
    max_cost_per_run: Decimal = Decimal("100")
    max_sample_concurrency: int = 4
    max_global_concurrency: int = 16
    call_timeout_ms: int = 120_000
    sample_timeout_ms: int = 300_000
    judge_reserved_calls: int = 1
    judge_reserved_tokens: int = 1000
    judge_reserved_cost: Decimal = Decimal("0.10")

    def __post_init__(self) -> None:
        integer_values = (
            self.max_nodes,
            self.max_depth,
            self.max_calls_per_sample,
            self.max_tokens_per_sample,
            self.max_calls_per_run,
            self.max_tokens_per_run,
            self.max_sample_concurrency,
            self.max_global_concurrency,
            self.call_timeout_ms,
            self.sample_timeout_ms,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("budget limits must be positive")
        if any(
            value < 0
            for value in (
                self.max_cost_per_sample,
                self.max_cost_per_run,
                self.judge_reserved_cost,
            )
        ):
            raise ValueError("cost limits must not be negative")


@dataclass(frozen=True)
class UsageDetails(ModelMixin):
    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    cost: Decimal | None = None
    cost_status: CostStatus = CostStatus.UNAVAILABLE
    provider: str | None = None
    model: str | None = None
    pricing_version: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


@dataclass(frozen=True)
class BudgetRequest(ModelMixin):
    calls: int = 1
    tokens: int = 0
    cost: Decimal = Decimal("0")
    required: bool = True
    purpose: str = "model"


class ReservationStatus(StrEnum):
    ACTIVE = "active"
    RECONCILED = "reconciled"
    RELEASED = "released"


@dataclass(frozen=True)
class BudgetReservation(ModelMixin):
    reservation_id: str
    sample_id: str
    request: BudgetRequest
    status: ReservationStatus = ReservationStatus.ACTIVE


@dataclass(frozen=True)
class BudgetSnapshot(ModelMixin):
    calls_used: int
    tokens_used: int
    cost_used: Decimal
    calls_reserved: int
    tokens_reserved: int
    cost_reserved: Decimal
    active_reservations: int
    unavailable_cost_calls: int = 0
