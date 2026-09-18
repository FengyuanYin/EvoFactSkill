"""Fail-closed lookup and immutable snapshots for label contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from evofact.core.label_models import DatasetLabelContract, LabelDefinition
from evofact.core.models import Sample


class LabelContractRegistry:
    def __init__(self, contracts: Iterable[DatasetLabelContract] = ()) -> None:
        self._contracts: dict[tuple[str, str], DatasetLabelContract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: DatasetLabelContract) -> None:
        if contract.key in self._contracts:
            raise ValueError(f"duplicate label contract: {contract.key!r}")
        self._contracts[contract.key] = contract

    def resolve(self, dataset_id: str, schema_id: str = "default") -> DatasetLabelContract:
        try:
            return self._contracts[(dataset_id, schema_id)]
        except KeyError as exc:
            raise KeyError(f"unknown label contract: {dataset_id}/{schema_id}") from exc

    def resolve_sample(self, sample: Sample) -> DatasetLabelContract:
        return self.resolve(sample.dataset, sample.label_schema_id)

    def contracts(self) -> tuple[DatasetLabelContract, ...]:
        return tuple(self._contracts[key] for key in sorted(self._contracts))

    @property
    def digest(self) -> str:
        payload = [{"key": item.key, "digest": item.digest} for item in self.contracts()]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def snapshot(self) -> "LabelContext":
        return LabelContext(self.contracts(), self.digest)


@dataclass(frozen=True)
class LabelContext:
    contracts: tuple[DatasetLabelContract, ...]
    registry_digest: str

    def registry(self) -> LabelContractRegistry:
        registry = LabelContractRegistry(self.contracts)
        if registry.digest != self.registry_digest:
            raise ValueError("label context digest does not match its contracts")
        return registry


def fixture_binary_contract(dataset_id: str = "fixture") -> DatasetLabelContract:
    """Explicit legacy fixture contract used only by offline examples and tests."""
    return DatasetLabelContract(
        dataset_id=dataset_id,
        schema_id="default",
        version="fixture-1",
        labels=(
            LabelDefinition("REAL", "The fixture claim is supported.", "Preserve support."),
            LabelDefinition("FAKE", "The fixture claim is refuted.", "Preserve refutation."),
        ),
        native_mapping=(("REAL", "REAL"), ("FAKE", "FAKE"), ("0", "REAL"), ("1", "FAKE")),
        positive_label="FAKE",
    )
