import csv
import json
from pathlib import Path
from typing import Iterable

from evofact.core.models import Sample
from evofact.data.base import DatasetDiagnostic


class AMTCeleAdapter:
    name = "amtcele"

    def discover(self, root: Path) -> DatasetDiagnostic:
        """函数作用：检查指定目录中是否存在当前适配器支持的数据文件。
        输入要求：`self` 应为已初始化的 `AMTCeleAdapter` 实例；`root`（Path）需符合函数签名约定。
        输出：返回 `DatasetDiagnostic` 类型结果；校验或下游调用失败时异常向上传递。"""
        files = tuple(
            str(p) for p in sorted(root.glob("*")) if p.suffix.casefold() in {".csv", ".jsonl"}
        )
        return DatasetDiagnostic(
            self.name, bool(files), files, "ready" if files else f"no AMTCele files under {root}"
        )

    def load(self, root: Path) -> Iterable[Sample]:
        """函数作用：从配置的存储位置读取并标准化当前对象负责的数据。
        输入要求：`self` 应为已初始化的 `AMTCeleAdapter` 实例；`root`（Path）需符合函数签名约定。
        输出：返回迭代器并逐项产出 `Iterable[Sample]` 所约定的结果；读取或解析失败时异常在迭代阶段抛出。"""
        diag = self.discover(root)
        if not diag.available:
            raise FileNotFoundError(diag.message)
        path = next(iter(sorted(root.glob("*.jsonl"))), None)
        if path:
            rows = [
                json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()
            ]
        else:
            path = next(iter(sorted(root.glob("*.csv"))))
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        for i, row in enumerate(rows):
            text = str(row.get("text") or row.get("content") or "").strip()
            if text:
                yield Sample(
                    str(row.get("id") or f"amtcele:{i}"),
                    self.name,
                    text,
                    row.get("label"),
                    str(row.get("domain") or "").rstrip("0123456789") or None,
                    metadata={"source_file": path.name},
                )
