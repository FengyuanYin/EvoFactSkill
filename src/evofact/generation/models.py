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
        """函数作用：在 `GenerationConfig` 数据类初始化后检查字段之间的业务约束。
        输入要求：`self` 应为已初始化的 `GenerationConfig` 实例；无其他显式输入。
        输出：返回 `None`；验证数据类字段，不满足约束时抛出 `ValueError`。"""
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
    """函数作用：负责当前模块中的 `numeric` 处理，封装调用方需要复用的业务步骤。
    输入要求：`value`（str）需符合函数签名约定。
    输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
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
        """函数作用：在 `EvidenceFact` 数据类初始化后检查字段之间的业务约束。
        输入要求：`self` 应为已初始化的 `EvidenceFact` 实例；无其他显式输入。
        输出：返回 `None`；验证数据类字段，不满足约束时抛出 `ValueError`。"""
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
        """函数作用：负责`EvidenceFact` 中的 `evidence_text` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `EvidenceFact` 实例；无其他显式输入。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        return f"{self.entity}的{self.attribute}为{numeric(self.value)}{self.unit}。"
