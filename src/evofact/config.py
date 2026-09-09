from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from evofact.generation.models import GenerationConfig


@dataclass(frozen=True)
class GateConfig:
    repeats: int = 3
    min_macro_f1_gain: float = 0.01
    min_coverage: float = 0.8
    max_protected_domain_drop: float = 0.02
    alpha: float = 0.05
    max_cost_ratio: float = 1.5

    def __post_init__(self) -> None:
        if self.repeats < 2 or not 0 <= self.min_coverage <= 1:
            raise ValueError("invalid gate configuration")


@dataclass(frozen=True)
class MetaLearningConfig:
    enabled: bool = False
    strategy: str = "repeated_holdout"
    episodes: int = 5
    meta_test_domain_count: int = 1
    min_source_domains: int = 3
    min_valid_episodes: int = 3
    min_mean_gain: float = 0.0
    max_negative_transfer_rate: float = 0.25
    max_worst_domain_drop: float = 0.1
    min_coverage: float = 0.5
    max_cost_ratio: float = 2.0
    max_calibration_increase: float = 0.2
    confidence_level: float = 0.95
    allow_specialization: bool = True
    aggregate_across_episodes: bool = True
    enforce_worst_domain: bool = True
    checkpoint_path: Path = Path("outputs/demse/checkpoint.json")

    def __post_init__(self) -> None:
        if self.strategy not in {"repeated_holdout", "leave_one_domain_out"}:
            raise ValueError("invalid meta-learning strategy")
        if self.episodes < 1 or self.meta_test_domain_count < 1:
            raise ValueError("episodes and meta_test_domain_count must be positive")
        if self.min_source_domains < 3 or self.min_valid_episodes < 1:
            raise ValueError("invalid domain or episode minimum")
        if not 0 <= self.max_negative_transfer_rate <= 1:
            raise ValueError("invalid negative transfer rate")
        if not 0 <= self.min_coverage <= 1 or not 0 < self.confidence_level < 1:
            raise ValueError("invalid meta-learning metric constraint")
        if self.max_calibration_increase < 0:
            raise ValueError("max_calibration_increase must be non-negative")


@dataclass(frozen=True)
class AppConfig:
    seed: int = 42
    output_dir: Path = Path("outputs")
    skill_store: Path = Path("skills/store")
    dataset_roots: dict[str, Path] = field(default_factory=dict)
    backend: str = "mock"
    model: str = "mock-v1"
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    meta_evolution: bool = False
    max_skills_per_item: int = 3
    gate: GateConfig = field(default_factory=GateConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        if self.backend not in {"mock", "openai-compatible"}:
            raise ValueError("backend must be mock or openai-compatible")
        if self.max_skills_per_item < 1:
            raise ValueError("max_skills_per_item must be positive")

    def resolved_api_key(self) -> str:
        value = os.getenv(self.api_key_env, "")
        if not value:
            raise ValueError(f"missing API key environment variable: {self.api_key_env}")
        return value


def _scalar(value: str) -> Any:
    value = value.strip()
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    if value.casefold() in {"null", "none"}:
        return None
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value.strip("\"'")


def _simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, sep, value = raw.strip().partition(":")
        if not sep:
            raise ValueError(f"invalid config line: {raw}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def load_config(path: str | Path) -> AppConfig:
    raw = _simple_yaml(Path(path).read_text(encoding="utf-8"))
    gate = GateConfig(**raw.pop("gate", {}))
    generation_raw = raw.pop("generation", {})
    if "store_path" in generation_raw:
        generation_raw["store_path"] = Path(generation_raw["store_path"])
    generation = GenerationConfig(**generation_raw)
    meta_raw = raw.pop("meta_learning", {})
    if "checkpoint_path" in meta_raw:
        meta_raw["checkpoint_path"] = Path(meta_raw["checkpoint_path"])
    meta_learning = MetaLearningConfig(**meta_raw)
    for key in ("output_dir", "skill_store"):
        if key in raw:
            raw[key] = Path(raw[key])
    raw["dataset_roots"] = {k: Path(v) for k, v in raw.get("dataset_roots", {}).items()}
    return AppConfig(gate=gate, meta_learning=meta_learning, generation=generation, **raw)
