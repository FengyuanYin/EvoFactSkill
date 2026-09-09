import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


@dataclass(frozen=True)
class GenerationConfig:
    enabled: bool = False
    proposer: str = "llm"
    batch_size: int = 8
    max_trace_examples: int = 32
    max_text_chars: int = 6000
    probe_fraction: float = 0.4
    store_path: Path = Path("outputs/adversarial-llm/generation-audit.json")

    def __post_init__(self):
        if self.proposer != "llm":
            raise ValueError("generation.proposer only supports llm")
        for value, lower, upper in (
            (self.batch_size, 2, 100),
            (self.max_trace_examples, 2, 200),
            (self.max_text_chars, 100, 30000),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("invalid generation budget")
        if not math.isfinite(self.probe_fraction) or not 0 < self.probe_fraction < 1:
            raise ValueError("invalid generation split")


def numeric(value: str) -> str:
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("fact value must be numeric") from exc
    if not number.is_finite() or abs(number) > Decimal("1e12"):
        raise ValueError("fact value outside supported range")
    return format(number.normalize(), "f")


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    sample_id: str
    domain: str
    event_id: str
    source: str
    entity: str
    attribute: str
    value: str
    unit: str
    verified: bool = False

    def __post_init__(self):
        if not all(
            isinstance(v, str) and v.strip()
            for v in (
                self.fact_id,
                self.sample_id,
                self.domain,
                self.event_id,
                self.source,
                self.entity,
                self.attribute,
                self.unit,
            )
        ):
            raise ValueError("fact provenance and slots must be non-empty strings")
        if any("\n" in v for v in (self.entity, self.attribute, self.unit)):
            raise ValueError("fact slots must be single-line")
        numeric(self.value)

    @property
    def evidence_text(self):
        return f"{self.entity}的{self.attribute}为{numeric(self.value)}{self.unit}。"
