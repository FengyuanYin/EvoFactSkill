from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from evofact.governance import GENERATION_AUDIT_SCHEMA_VERSION

from .budget_models import UsageDetails
from .models import ModelMixin, Sample


@dataclass(frozen=True)
class GenerationLineage(ModelMixin):
    generated_sample_id: str
    source_sample_id: str
    lineage_id: str
    episode_id: str
    strategy: str
    event_id: str | None
    evidence_digests: tuple[str, ...]
    generator_package_digest: str
    model: str
    config_digest: str
    created_at: str
    label_schema_id: str = "legacy-binary"
    label_contract_digest: str = ""


@dataclass(frozen=True)
class GeneratedSample(ModelMixin):
    sample: Sample
    lineage: GenerationLineage
    synthetic: bool = True


class VerificationStatus(StrEnum):
    CLASSIFIED = "classified"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class VerificationDecision(ModelMixin):
    sample_id: str
    status: VerificationStatus
    predicted_label: str | None
    accepted: bool
    reason: str
    verifier_version: str
    label_contract_digest: str = ""


@dataclass(frozen=True)
class GenerationAuditEntry(ModelMixin):
    audit_id: str
    lineage: GenerationLineage
    accepted: bool
    reasons: tuple[str, ...]
    request_digest: str
    response_digest: str
    verifier: VerificationDecision | None
    usage: UsageDetails = field(default_factory=UsageDetails)
    metrics: dict[str, float] = field(default_factory=dict)
    schema_version: str = GENERATION_AUDIT_SCHEMA_VERSION


@dataclass(frozen=True)
class GenerationBatch(ModelMixin):
    episode_id: str
    generator_package_digest: str
    samples: tuple[GeneratedSample, ...]
    request_digest: str
    response_digest: str
    usage: UsageDetails = field(default_factory=UsageDetails)
    label_contract_digest: str = ""


@dataclass(frozen=True)
class VerificationBatch(ModelMixin):
    episode_id: str
    decisions: tuple[VerificationDecision, ...]
    verifier_version: str
    usage: UsageDetails = field(default_factory=UsageDetails)
    label_contract_digest: str = ""


@dataclass(frozen=True)
class GenerationMetrics(ModelMixin):
    validity_rate: float
    agreement_rate: float
    evidence_rate: float
    diversity: float
    duplicate_rate: float
    difficulty: float
    teaching_gain: float
    leakage_rate: float
    safety_rejection_rate: float
    cost: Decimal | None = None
    label_coverage_rate: float = 0.0
    worst_label_agreement: float = 0.0
    per_label_acceptance: dict[str, float] = field(default_factory=dict)
    authentic_macro_f1: float = 0.0
    worst_label_probe_drop: float = 0.0
    label_schema_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedGeneratorEvaluation(ModelMixin):
    source_episode_id: str
    evaluation_episode_id: str
    champion_digest: str
    challenger_digest: str
    champion: GenerationMetrics
    challenger: GenerationMetrics
    metadata: dict[str, Any] = field(default_factory=dict)
    label_contract_digest: str = ""
