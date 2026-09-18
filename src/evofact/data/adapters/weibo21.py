import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from evofact.core.label_models import DatasetLabelContract, LabelDefinition
from evofact.core.models import Sample
from evofact.data.base import DatasetDiagnostic
from evofact.data.deduplication import stable_sample_id


class Weibo21Adapter:
    name = "weibo21"
    filename = "weibo21_all.jsonl"

    def label_contracts(self) -> tuple[DatasetLabelContract, ...]:
        return (
            DatasetLabelContract(
                dataset_id=self.name,
                schema_id="weibo21-binary-v1",
                version="1",
                labels=(
                    LabelDefinition(
                        "REAL",
                        "The claim is supported as factual.",
                        "Preserve a supported factual claim.",
                    ),
                    LabelDefinition(
                        "FAKE",
                        "The claim is contradicted or fabricated.",
                        "Create a contradicted or fabricated claim while keeping it auditable.",
                    ),
                ),
                native_mapping=(("0", "REAL"), ("1", "FAKE"), ("REAL", "REAL"), ("FAKE", "FAKE")),
                positive_label="FAKE",
            ),
        )

    def discover(self, root: Path) -> DatasetDiagnostic:
        """函数作用：检查指定目录中是否存在当前适配器支持的数据文件。
        输入要求：`self` 应为已初始化的 `Weibo21Adapter` 实例；`root`（Path）需符合函数签名约定。
        输出：返回 `DatasetDiagnostic` 类型结果；校验或下游调用失败时异常向上传递。"""
        path = root / self.filename
        files = (str(path),) if path.is_file() else ()
        return DatasetDiagnostic(
            self.name,
            bool(files),
            files,
            "ready" if files else f"missing {self.filename} under {root}",
        )

    def _rows(self, path: Path) -> Iterator[dict[str, Any]]:
        """函数作用：负责`Weibo21Adapter` 中的 `_rows` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `Weibo21Adapter` 实例；`path`（Path）需符合函数签名约定。
        输出：逐行返回 JSON 对象；校验或下游调用失败时异常向上传递。"""
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                yield value

    def load(self, root: Path) -> Iterable[Sample]:
        """函数作用：从配置的存储位置读取并标准化当前对象负责的数据。
        输入要求：`self` 应为已初始化的 `Weibo21Adapter` 实例；`root`（Path）需符合函数签名约定。
        输出：返回迭代器并逐项产出 `Iterable[Sample]` 所约定的结果；读取或解析失败时异常在迭代阶段抛出。"""
        diag = self.discover(root)
        if not diag.available:
            raise FileNotFoundError(diag.message)
        path = root / self.filename
        samples = []
        for row in self._rows(path):
            text = str(row.get("content") or row.get("text") or "").strip()
            domain = str(row.get("category") or "").strip() or None
            if text and domain:
                samples.append(
                    Sample(
                        stable_sample_id(self.name, text),
                        self.name,
                        text,
                        row.get("label"),
                        domain,
                        row.get("event_id"),
                        metadata={
                            "source_split": row.get("split"),
                            "source_file": path.name,
                        },
                        label_schema_id="weibo21-binary-v1",
                    )
                )
        yield from samples
