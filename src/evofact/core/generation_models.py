from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

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


@dataclass(frozen=True)
class GeneratedSample(ModelMixin):
    sample: Sample
    lineage: GenerationLineage
    synthetic: bool = True


@dataclass(frozen=True)
class VerificationDecision(ModelMixin):
    sample_id: str
    label: Literal["REAL", "FAKE", "UNKNOWN"]
    accepted: bool
    reason: str
    verifier_version: str


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


@dataclass(frozen=True)
class VerificationBatch(ModelMixin):
    episode_id: str
    decisions: tuple[VerificationDecision, ...]
    verifier_version: str
    usage: UsageDetails = field(default_factory=UsageDetails)


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


@dataclass(frozen=True)
class PairedGeneratorEvaluation(ModelMixin):
    source_episode_id: str
    evaluation_episode_id: str
    champion_digest: str
    challenger_digest: str
    champion: GenerationMetrics
    challenger: GenerationMetrics
    metadata: dict[str, Any] = field(default_factory=dict)
