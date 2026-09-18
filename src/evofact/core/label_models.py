"""Versioned, dataset-owned label contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any

RUNTIME_ABSTAIN_LABEL = "ABSTAIN"


class DecisionOrigin(StrEnum):
    JUDGE = "judge"
    RUNTIME = "runtime"
    LEGACY = "legacy"


@dataclass(frozen=True)
class LabelDefinition:
    name: str
    description: str
    generation_guidance: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("label name must not be empty")
        if self.name.casefold() == RUNTIME_ABSTAIN_LABEL.casefold():
            raise ValueError("ABSTAIN is reserved for runtime failures")
        if not self.description.strip():
            raise ValueError(f"label {self.name!r} requires a description")
        if not self.generation_guidance.strip():
            raise ValueError(f"label {self.name!r} requires generation guidance")

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return asdict(self)


@dataclass(frozen=True)
class DatasetLabelContract:
    dataset_id: str
    schema_id: str
    version: str
    labels: tuple[LabelDefinition, ...]
    native_mapping: tuple[tuple[str, str], ...]
    positive_label: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.schema_id.strip() or not self.version.strip():
            raise ValueError("dataset_id, schema_id, and version must not be empty")
        if len(self.labels) < 2:
            raise ValueError("a label contract requires at least two labels")
        names = self.allowed_labels
        if len(set(names)) != len(names) or len({name.casefold() for name in names}) != len(names):
            raise ValueError("canonical label names must be unique")
        mapping: dict[str, str] = {}
        for native, canonical in self.native_mapping:
            key = self._native_key(native)
            if not key:
                raise ValueError("native label keys must not be empty")
            if key in mapping and mapping[key] != canonical:
                raise ValueError(f"conflicting native label mapping for {native!r}")
            if canonical not in names:
                raise ValueError(f"mapping target {canonical!r} is not an allowed label")
            mapping[key] = canonical
        if not mapping:
            raise ValueError("native_mapping must not be empty")
        if self.positive_label is not None and self.positive_label not in names:
            raise ValueError("positive_label must be an allowed label")
        if self.positive_label is not None and len(names) != 2:
            raise ValueError("positive_label is only supported by binary contracts")

    @staticmethod
    def _native_key(value: object) -> str:
        return str(value).strip().casefold()

    @property
    def key(self) -> tuple[str, str]:
        return (self.dataset_id, self.schema_id)

    @property
    def allowed_labels(self) -> tuple[str, ...]:
        return tuple(label.name for label in self.labels)

    @property
    def digest(self) -> str:
        payload = {
            "dataset_id": self.dataset_id,
            "schema_id": self.schema_id,
            "version": self.version,
            "labels": [asdict(label) for label in self.labels],
            "native_mapping": list(self.native_mapping),
            "positive_label": self.positive_label,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def normalize(self, value: object) -> str:
        if value is None:
            raise ValueError(f"missing native label for {self.dataset_id}/{self.schema_id}")
        mapping = {self._native_key(native): canonical for native, canonical in self.native_mapping}
        try:
            return mapping[self._native_key(value)]
        except KeyError as exc:
            raise ValueError(
                f"unknown native label {value!r} for {self.dataset_id}/{self.schema_id}"
            ) from exc

    def require_label(self, value: object) -> str:
        label = str(value).strip()
        if label not in self.allowed_labels:
            raise ValueError(
                f"label {label!r} is outside {self.dataset_id}/{self.schema_id}: "
                f"{self.allowed_labels!r}"
            )
        return label

    def prompt_view(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "schema_id": self.schema_id,
            "version": self.version,
            "allowed_labels": list(self.allowed_labels),
            "labels": [
                {"name": item.name, "description": item.description} for item in self.labels
            ],
        }

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        payload = asdict(self)
        payload["allowed_labels"] = self.allowed_labels
        payload["digest"] = self.digest
        return payload
