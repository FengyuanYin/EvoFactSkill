from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evofact.generation.models import GenerationConfig


@dataclass(frozen=True)
class GateConfig:  # 验证门配置
    repeats: int = 3
    min_macro_f1_gain: float = 0.01
    min_coverage: float = 0.8
    max_protected_domain_drop: float = 0.02
    alpha: float = 0.05
    max_cost_ratio: float = 1.5

    def __post_init__(self) -> None:
        """函数作用：在 `GateConfig` 数据类初始化后检查字段之间的业务约束。
        输入要求：`self` 应为已初始化的 `GateConfig` 实例；无其他显式输入。
        输出：返回 `None`；验证数据类字段，不满足约束时抛出 `ValueError`。"""
        if self.repeats < 2 or not 0 <= self.min_coverage <= 1:
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


@dataclass(frozen=True)
class AppConfig:  # runner 配置
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
        """函数作用：在 `AppConfig` 数据类初始化后检查字段之间的业务约束。
        输入要求：`self` 应为已初始化的 `AppConfig` 实例；无其他显式输入。
        输出：返回 `None`；验证数据类字段，不满足约束时抛出 `ValueError`。"""
        if self.backend not in {"mock", "openai-compatible"}:
            raise ValueError("backend must be mock or openai-compatible")
        if self.max_skills_per_item < 1:
            raise ValueError("max_skills_per_item must be positive")

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
