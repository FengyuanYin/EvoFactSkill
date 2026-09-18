from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from evofact.generation.models import GenerationConfig


@dataclass(frozen=True)
class DAGConfig:
    max_nodes: int = 8
    max_depth: int = 4
    strict_legacy_order: bool = False
    node_timeout_ms: int = 120_000
    sample_timeout_ms: int = 300_000

    def __post_init__(self) -> None:
        if min(self.max_nodes, self.max_depth, self.node_timeout_ms, self.sample_timeout_ms) < 1:
            raise ValueError("DAG limits must be positive")


@dataclass(frozen=True)
class BudgetConfig:
    max_calls_per_sample: int = 8
    max_tokens_per_sample: int = 12000
    max_cost_per_sample: Decimal = Decimal("1")
    max_calls_per_run: int = 1000
    max_tokens_per_run: int = 1_000_000
    max_cost_per_run: Decimal = Decimal("100")
    max_sample_concurrency: int = 4
    max_global_concurrency: int = 16
    judge_reserved_calls: int = 1
    judge_reserved_tokens: int = 1000
    judge_reserved_cost: Decimal = Decimal("0.10")

    def __post_init__(self) -> None:
        integer_values = (
            self.max_calls_per_sample,
            self.max_tokens_per_sample,
            self.max_calls_per_run,
            self.max_tokens_per_run,
            self.max_sample_concurrency,
            self.max_global_concurrency,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("budget limits must be positive")


@dataclass(frozen=True)
class PricingConfig:
    provider: str = "openai-compatible"
    table_path: Path | None = None
    require_cost_for_promotion: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    batch_size: int = 16
    max_concurrent_samples: int = 4
    progress: str = "auto"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.max_concurrent_samples < 1:
            raise ValueError("execution batch size and sample concurrency must be positive")
        if self.progress not in {"auto", "on", "off"}:
            raise ValueError("execution.progress must be auto, on, or off")


@dataclass(frozen=True)
class GateConfig:  # 验证门配置
    repeats: int = 3
    min_macro_f1_gain: float = 0.01
    min_coverage: float = 0.8
    max_protected_domain_drop: float = 0.02
    alpha: float = 0.05
    max_cost_ratio: float = 1.5
    max_calibration_increase: float = 0.2

    def __post_init__(self) -> None:
        """函数作用：在 `GateConfig` 数据类初始化后检查字段之间的业务约束。
        输入要求：`self` 应为已初始化的 `GateConfig` 实例；无其他显式输入。
        输出：返回 `None`；验证数据类字段，不满足约束时抛出 `ValueError`。"""
        if self.repeats < 2 or not 0 <= self.min_coverage <= 1 or self.max_calibration_increase < 0:
            raise ValueError("invalid gate configuration")


@dataclass(frozen=True)
class MetaLearningConfig:  # 元学习配置
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
        """函数作用：在 `MetaLearningConfig` 数据类初始化后检查字段之间的业务约束。
        输入要求：`self` 应为已初始化的 `MetaLearningConfig` 实例；无其他显式输入。
        输出：返回 `None`；验证数据类字段，不满足约束时抛出 `ValueError`。"""
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


@dataclass
class EvolutionConfig:
    proposer: str = "rule"

    fallback_to_rule: bool = True

    optimizer_skill: str = "skill_optimizer"

    max_reports: int = 12

    max_skills: int = 12

    max_text_chars: int = 8000

    max_instructions_chars: int = 60000

    max_total_chars: int = 300000

    def __post_init__(self) -> None:
        if self.proposer not in {"rule", "llm"}:
            raise ValueError("evolution.proposer must be rule or llm")

        if not self.optimizer_skill.strip():
            raise ValueError("evolution.optimizer_skill must not be empty")

        limits = {
            "max_reports": self.max_reports,
            "max_skills": self.max_skills,
            "max_text_chars": self.max_text_chars,
            "max_instructions_chars": self.max_instructions_chars,
            "max_total_chars": self.max_total_chars,
        }

        invalid = [name for name, value in limits.items() if value < 1]

        if invalid:
            raise ValueError("evolution limits must be positive: " + ", ".join(invalid))


@dataclass(frozen=True)
class DataConfig:
    dataset: str | None = None
    root: Path | None = None
    train_domains: tuple[str, ...] = ()
    final_test_domains: tuple[str, ...] = ()
    excluded_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        configured = bool(
            self.dataset
            or self.root
            or self.train_domains
            or self.final_test_domains
            or self.excluded_domains
        )
        if not configured:
            return
        if not self.dataset or self.root is None:
            raise ValueError("data.dataset and data.root must be configured together")
        if not self.train_domains or not self.final_test_domains:
            raise ValueError("data.train_domains and data.final_test_domains must be non-empty")

        train = set(self.train_domains)
        final = set(self.final_test_domains)
        excluded = set(self.excluded_domains)
        if len(train) != len(self.train_domains):
            raise ValueError("data.train_domains must not contain duplicates")
        if len(final) != len(self.final_test_domains):
            raise ValueError("data.final_test_domains must not contain duplicates")
        if len(excluded) != len(self.excluded_domains):
            raise ValueError("data.excluded_domains must not contain duplicates")
        if train & final or train & excluded or final & excluded:
            raise ValueError("train, final-test and excluded domains must be disjoint")


@dataclass(frozen=True)
class AppConfig:  # runner 配置
    seed: int = 42
    output_dir: Path = Path("outputs")
    skill_store: Path = Path("skills/store")
    dataset_roots: dict[str, Path] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    backend: str = "mock"
    model: str = "mock-v1"
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    routing_strategy: str = "utility-aware"
    meta_evolution: bool = False
    max_skills_per_item: int = 3
    gate: GateConfig = field(default_factory=GateConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    dag: DAGConfig = field(default_factory=DAGConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self) -> None:
        """函数作用：在 `AppConfig` 数据类初始化后检查字段之间的业务约束。
        输入要求：`self` 应为已初始化的 `AppConfig` 实例；无其他显式输入。
        输出：返回 `None`；验证数据类字段，不满足约束时抛出 `ValueError`。"""
        if self.backend not in {"mock", "openai-compatible"}:
            raise ValueError("backend must be mock or openai-compatible")
        if self.max_skills_per_item < 1:
            raise ValueError("max_skills_per_item must be positive")
        if self.routing_strategy not in {
            "utility-aware",
            "all-experts",
            "random",
            "static",
            "llm",
        }:
            raise ValueError(
                "routing_strategy must be utility-aware, all-experts, random, static, or llm"
            )

    def resolved_api_key(self) -> str:
        """函数作用：负责`AppConfig` 中的 `resolved_api_key` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `AppConfig` 实例；无其他显式输入。
        输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
        value = os.getenv(self.api_key_env, "")
        if not value:
            raise ValueError(f"missing API key environment variable: {self.api_key_env}")
        return value


def _scalar(value: str) -> Any:
    """函数作用：负责当前模块中的 `_scalar` 处理，封装调用方需要复用的业务步骤。
    输入要求：`value`（str）需符合函数签名约定。
    输出：返回 `Any` 类型结果；校验或下游调用失败时异常向上传递。"""
    value = value.strip()
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    if value.casefold() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("config sequence must be a JSON-style list")
        return parsed
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value.strip("\"'")


def _simple_yaml(text: str) -> dict[str, Any]:
    """函数作用：负责当前模块中的 `_simple_yaml` 处理，封装调用方需要复用的业务步骤。
    输入要求：`text`（str）需符合函数签名约定。
    输出：返回 `dict[str, Any]` 类型结果；校验或下游调用失败时异常向上传递。"""
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
    """函数作用：读取简化 YAML 配置并构造经过校验的应用配置对象。
    输入要求：`path`（str | Path）需符合函数签名约定。
    输出：返回 `AppConfig` 类型结果；校验或下游调用失败时异常向上传递。"""
    raw = _simple_yaml(Path(path).read_text(encoding="utf-8"))
    data_raw = raw.pop("data", {})
    if "root" in data_raw and data_raw["root"] is not None:
        data_raw["root"] = Path(data_raw["root"])
    for key in ("train_domains", "final_test_domains", "excluded_domains"):
        if key in data_raw:
            value = data_raw[key]
            if isinstance(value, str):
                value = [item.strip() for item in value.split(",") if item.strip()]
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"data.{key} must be a list")
            data_raw[key] = tuple(str(item).strip() for item in value if str(item).strip())
    data = DataConfig(**data_raw)
    gate = GateConfig(**raw.pop("gate", {}))
    evolution_raw = raw.pop("evolution", {})
    evolution = EvolutionConfig(**evolution_raw)
    generation_raw = raw.pop("generation", {})
    if "store_path" in generation_raw:
        generation_raw["store_path"] = Path(generation_raw["store_path"])
    generation = GenerationConfig(**generation_raw)
    dag = DAGConfig(**raw.pop("dag", {}))
    budget_raw = raw.pop("budget", {})
    for key in ("max_cost_per_sample", "max_cost_per_run", "judge_reserved_cost"):
        if key in budget_raw:
            budget_raw[key] = Decimal(str(budget_raw[key]))
    budget = BudgetConfig(**budget_raw)
    pricing_raw = raw.pop("pricing", {})
    if pricing_raw.get("table_path") is not None:
        pricing_raw["table_path"] = Path(pricing_raw["table_path"])
    pricing = PricingConfig(**pricing_raw)
    execution = ExecutionConfig(**raw.pop("execution", {}))
    meta_raw = raw.pop("meta_learning", {})
    if "checkpoint_path" in meta_raw:
        meta_raw["checkpoint_path"] = Path(meta_raw["checkpoint_path"])
    meta_learning = MetaLearningConfig(**meta_raw)
    for key in ("output_dir", "skill_store"):
        if key in raw:
            raw[key] = Path(raw[key])
    raw["dataset_roots"] = {k: Path(v) for k, v in raw.get("dataset_roots", {}).items()}
    return AppConfig(
        gate=gate,
        meta_learning=meta_learning,
        generation=generation,
        evolution=evolution,
        dag=dag,
        budget=budget,
        pricing=pricing,
        execution=execution,
        data=data,
        **raw,
    )
