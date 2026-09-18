import csv
from pathlib import Path
from typing import Iterable

from evofact.core.label_models import DatasetLabelContract, LabelDefinition
from evofact.core.models import Sample
from evofact.data.base import DatasetDiagnostic


class AdvFakeAdapter:
    name = "advfake"

    def label_contracts(self) -> tuple[DatasetLabelContract, ...]:
        return (
            DatasetLabelContract(
                dataset_id=self.name,
                schema_id="advfake-binary-v1",
                version="1",
                labels=(
                    LabelDefinition(
                        "REAL",
                        "The claim remains factually supported.",
                        "Preserve the supported meaning under surface-form changes.",
                    ),
                    LabelDefinition(
                        "FAKE",
                        "The claim is adversarially falsified or unsupported.",
                        "Construct an auditable adversarial falsehood.",
                    ),
                ),
                native_mapping=(("REAL", "REAL"), ("FAKE", "FAKE"), ("0", "REAL"), ("1", "FAKE")),
                positive_label="FAKE",
            ),
        )

    def discover(self, root: Path) -> DatasetDiagnostic:
        """函数作用：检查指定目录中是否存在当前适配器支持的数据文件。
        输入要求：`self` 应为已初始化的 `AdvFakeAdapter` 实例；`root`（Path）需符合函数签名约定。
        输出：返回 `DatasetDiagnostic` 类型结果；校验或下游调用失败时异常向上传递。"""
        files = tuple(str(p) for p in sorted(root.glob("*.csv")))
        return DatasetDiagnostic(
            self.name, bool(files), files, "ready" if files else f"no AdvFake CSV under {root}"
        )

    def load(self, root: Path) -> Iterable[Sample]:
        """函数作用：从配置的存储位置读取并标准化当前对象负责的数据。
        输入要求：`self` 应为已初始化的 `AdvFakeAdapter` 实例；`root`（Path）需符合函数签名约定。
        输出：返回迭代器并逐项产出 `Iterable[Sample]` 所约定的结果；读取或解析失败时异常在迭代阶段抛出。"""
        diag = self.discover(root)
        if not diag.available:
            raise FileNotFoundError(diag.message)
        path = Path(diag.files[0])
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for i, row in enumerate(rows):
            pair = str(row.get("id") or f"advfake:{i}")
            explicit = str(
                row.get("adversarial") or row.get("adversarial_text") or row.get("text") or ""
            ).strip()
            source_text = str(row.get("title") or "").strip()
            adversarial_text = str(row.get("f_title") or "").strip()
            variants = (
                [(explicit, row.get("label"), "adversarial")]
                if explicit
                else [(source_text, "REAL", "source"), (adversarial_text, "FAKE", "adversarial")]
            )
            for text, label, variant in variants:
                if not text:
                    continue
                yield Sample(
                    f"{pair}:{variant}",
                    self.name,
                    text,
                    label,
                    event_id=pair,
                    metadata={
                        "original_text": row.get("original")
                        or row.get("original_text")
                        or source_text,
                        "attack_type": row.get("attack_type"),
                        "variant": variant,
                        "source_file": path.name,
                        "split_role": "test",
                        "robustness_only": True,
                    },
                    label_schema_id="advfake-binary-v1",
                )
