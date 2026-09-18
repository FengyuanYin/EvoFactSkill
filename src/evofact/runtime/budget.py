from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal

from evofact.core.budget_models import (
    BudgetLimits,
    BudgetRequest,
    BudgetReservation,
    BudgetSnapshot,
    UsageDetails,
)
from evofact.governance.budget_policy import evaluate_budget


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class _Ledger:
    calls_used: int = 0
    tokens_used: int = 0
    cost_used: Decimal = Decimal("0")
    calls_reserved: int = 0
    tokens_reserved: int = 0
    cost_reserved: Decimal = Decimal("0")
    unavailable_cost_calls: int = 0

    def snapshot(self, active: int) -> BudgetSnapshot:
        return BudgetSnapshot(
            self.calls_used,
            self.tokens_used,
            self.cost_used,
            self.calls_reserved,
            self.tokens_reserved,
            self.cost_reserved,
            active,
            self.unavailable_cost_calls,
        )


class BudgetManager:
    def __init__(self, limits: BudgetLimits = BudgetLimits()):
        self.limits = limits
        self._run = _Ledger()
        self._samples: dict[str, _Ledger] = {}
        self._reservations: dict[str, BudgetReservation] = {}
        self._lock = asyncio.Lock()
        self._global = asyncio.Semaphore(limits.max_global_concurrency)
        self._sample_semaphores: dict[str, asyncio.Semaphore] = {}

    def _sample(self, sample_id: str) -> _Ledger:
        return self._samples.setdefault(sample_id, _Ledger())

    async def reserve(
        self, sample_id: str, request: BudgetRequest, *, judge: bool = False
    ) -> BudgetReservation:
        async with self._lock:
            sample = self._sample(sample_id)
            active = len(self._reservations)
            decision = evaluate_budget(
                request,
                sample.snapshot(active),
                self._run.snapshot(active),
                self.limits,
                judge=judge,
            )
            if not decision.allowed:
                raise BudgetExceeded(decision.reason)
            reservation = BudgetReservation(uuid.uuid4().hex, sample_id, request)
            self._reservations[reservation.reservation_id] = reservation
            for ledger in (sample, self._run):
                ledger.calls_reserved += request.calls
                ledger.tokens_reserved += request.tokens
                ledger.cost_reserved += request.cost
            return reservation

    async def reconcile(self, reservation: BudgetReservation, usage: UsageDetails) -> None:
        async with self._lock:
            current = self._reservations.pop(reservation.reservation_id, None)
            if current is None:
                raise ValueError("reservation is not active")
            sample = self._sample(current.sample_id)
            actual_cost = usage.cost or Decimal("0")
            for ledger in (sample, self._run):
                ledger.calls_reserved -= current.request.calls
                ledger.tokens_reserved -= current.request.tokens
                ledger.cost_reserved -= current.request.cost
                ledger.calls_used += usage.calls
                ledger.tokens_used += usage.total_tokens
                ledger.cost_used += actual_cost
                if usage.calls and usage.cost is None:
                    ledger.unavailable_cost_calls += usage.calls

    async def release(self, reservation: BudgetReservation) -> None:
        async with self._lock:
            current = self._reservations.pop(reservation.reservation_id, None)
            if current is None:
                return
            sample = self._sample(current.sample_id)
            for ledger in (sample, self._run):
                ledger.calls_reserved -= current.request.calls
                ledger.tokens_reserved -= current.request.tokens
                ledger.cost_reserved -= current.request.cost

    def snapshot(self, sample_id: str | None = None) -> BudgetSnapshot:
        ledger = self._run if sample_id is None else self._sample(sample_id)
        return ledger.snapshot(len(self._reservations))

    def export_state(self) -> dict:
        """Return a checkpoint-safe ledger without transient reservations."""
        if self._reservations:
            raise ValueError("cannot checkpoint a budget with active reservations")
        return {
            "run": self._run.snapshot(0).model_dump(),
            "samples": {
                sample_id: ledger.snapshot(0).model_dump()
                for sample_id, ledger in sorted(self._samples.items())
            },
        }

    def restore(self, state: dict, *, allow_forward: bool = False) -> None:
        """Restore reconciled usage before resumed work is dispatched."""
        if self._reservations:
            raise ValueError("cannot restore over active budget reservations")

        def ledger(raw: dict) -> _Ledger:
            snapshot = BudgetSnapshot(
                calls_used=int(raw.get("calls_used", 0)),
                tokens_used=int(raw.get("tokens_used", 0)),
                cost_used=Decimal(str(raw.get("cost_used", "0"))),
                calls_reserved=int(raw.get("calls_reserved", 0)),
                tokens_reserved=int(raw.get("tokens_reserved", 0)),
                cost_reserved=Decimal(str(raw.get("cost_reserved", "0"))),
                active_reservations=int(raw.get("active_reservations", 0)),
                unavailable_cost_calls=int(raw.get("unavailable_cost_calls", 0)),
            )
            numeric = (
                snapshot.calls_used,
                snapshot.tokens_used,
                snapshot.cost_used,
                snapshot.calls_reserved,
                snapshot.tokens_reserved,
                snapshot.cost_reserved,
                snapshot.active_reservations,
                snapshot.unavailable_cost_calls,
            )
            if any(value < 0 for value in numeric):
                raise ValueError("budget checkpoint contains negative values")
            if (
                snapshot.active_reservations
                or snapshot.calls_reserved
                or snapshot.tokens_reserved
                or snapshot.cost_reserved
            ):
                raise ValueError("budget checkpoint contains transient reservations")
            return _Ledger(
                calls_used=snapshot.calls_used,
                tokens_used=snapshot.tokens_used,
                cost_used=snapshot.cost_used,
                unavailable_cost_calls=snapshot.unavailable_cost_calls,
            )

        restored_run = ledger(dict(state.get("run", {})))
        restored_samples = {
            str(sample_id): ledger(dict(raw))
            for sample_id, raw in dict(state.get("samples", {})).items()
        }
        current = self.snapshot()
        if current.calls_used or current.tokens_used or current.cost_used:
            if self._run != restored_run or self._samples != restored_samples:

                def precedes(before: _Ledger, after: _Ledger) -> bool:
                    return (
                        before.calls_used <= after.calls_used
                        and before.tokens_used <= after.tokens_used
                        and before.cost_used <= after.cost_used
                        and before.unavailable_cost_calls <= after.unavailable_cost_calls
                    )

                forward = (
                    allow_forward
                    and precedes(self._run, restored_run)
                    and all(
                        sample_id in restored_samples
                        and precedes(ledger, restored_samples[sample_id])
                        for sample_id, ledger in self._samples.items()
                    )
                )
                if not forward:
                    raise ValueError("cannot replace a non-empty budget ledger")
                self._run = restored_run
                self._samples = restored_samples
            return
        self._run = restored_run
        self._samples = restored_samples

    @asynccontextmanager
    async def concurrency(self, sample_id: str):
        sample_semaphore = self._sample_semaphores.setdefault(
            sample_id,
            asyncio.Semaphore(self.limits.max_sample_concurrency),
        )
        async with self._global:
            async with sample_semaphore:
                yield

    @asynccontextmanager
    async def call(self, sample_id: str, request: BudgetRequest, *, judge: bool = False):
        reservation = await self.reserve(sample_id, request, judge=judge)
        reconciled = False
        try:
            async with self.concurrency(sample_id):
                yield reservation
                reconciled = reservation.reservation_id not in self._reservations
        finally:
            if not reconciled:
                await self.release(reservation)
