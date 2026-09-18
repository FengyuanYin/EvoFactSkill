import json
from dataclasses import asdict
from pathlib import Path

from evofact.core.models import InferenceTrace


class TraceStore:
    def __init__(self, path: Path):
        """函数作用：创建并初始化 `TraceStore` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `TraceStore` 实例；`path`（Path）需符合函数签名约定。
        输出：返回 `None`；初始化 `TraceStore` 的实例状态，构造参数非法时可能抛出异常。"""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: InferenceTrace) -> None:
        """函数作用：把一条新记录追加到持久化存储，并维护对象内的完成状态。
        输入要求：`self` 应为已初始化的 `TraceStore` 实例；`trace`（InferenceTrace）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责修改对象状态或持久化文件。"""
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(trace), ensure_ascii=False, default=str) + "\n")

    def read_raw(self, *, allow_v1: bool = False) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            version = row.get("schema_version", "inference_trace_v1")
            if version == "inference_trace_v1" and not allow_v1:
                raise ValueError("trace v1 requires explicit compatibility adapter")
            if version not in {
                "inference_trace_v1",
                "inference_trace_v2",
                "inference_trace_v3",
                "inference_trace_v4",
            }:
                raise ValueError(f"unsupported trace schema: {version}")
        return rows


def adapt_trace_v1(row: dict) -> dict:
    if row.get("schema_version", "inference_trace_v1") != "inference_trace_v1":
        raise ValueError("adapter accepts only trace v1")
    return {
        **row,
        "schema_version": "inference_trace_v2",
        "execution_plan": None,
        "node_executions": [],
        "execution_summary": None,
        "resource_versions": {},
        "compatibility_adapter": "trace_v1_to_v2",
    }
