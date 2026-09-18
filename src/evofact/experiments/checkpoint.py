import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from evofact.governance import META_CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True)
class CheckpointIdentity:
    config_digest: str
    real_data_digest: str
    split_digest: str
    lineage_digest: str
    package_bank_digest: str
    generator_package_digest: str | None
    governance_digest: str
    dag_policy_digest: str
    budget_policy_digest: str
    pricing_version: str
    episode_plan_digest: str
    label_contract_digest: str = "legacy"


class CheckpointV2Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, identity: CheckpointIdentity, state: dict) -> None:
        payload = {
            "schema_version": META_CHECKPOINT_SCHEMA_VERSION,
            "identity": asdict(identity),
            "state": state,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".checkpoint-v2-", suffix=".json"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    payload, stream, ensure_ascii=False, sort_keys=True, indent=2, default=str
                )
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, identity: CheckpointIdentity) -> dict:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != META_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint v1 may be read for reports but cannot resume as v2")
        expected = asdict(identity)
        for key, value in expected.items():
            if payload.get("identity", {}).get(key) != value:
                raise ValueError(f"checkpoint identity mismatch: {key}")
        return payload.get("state", {})


class Checkpoint:
    def __init__(self, path: Path):
        """函数作用：创建并初始化 `Checkpoint` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `Checkpoint` 实例；`path`（Path）需符合函数签名约定。
        输出：返回 `None`；初始化 `Checkpoint` 的实例状态，构造参数非法时可能抛出异常。"""
        self.path = Path(path)
        self.completed = set()

    def load(self):
        """函数作用：从配置的存储位置读取并标准化当前对象负责的数据。
        输入要求：`self` 应为已初始化的 `Checkpoint` 实例；无其他显式输入。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        if self.path.exists():
            self.completed = {
                json.loads(x)["sample_id"]
                for x in self.path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            }
        return self.completed

    def append(self, sample_id: str, payload: dict):
        """函数作用：把一条新记录追加到持久化存储，并维护对象内的完成状态。
        输入要求：`self` 应为已初始化的 `Checkpoint` 实例；`sample_id`（str）需符合函数签名约定；`payload`（dict）需符合函数签名约定。
        输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        if sample_id in self.completed:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"sample_id": sample_id, **payload}, ensure_ascii=False, default=str)
                + "\n"
            )
        self.completed.add(sample_id)
        return True


class MetaCheckpointStore:
    def __init__(self, path: Path):
        """函数作用：创建并初始化 `MetaCheckpointStore` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `MetaCheckpointStore` 实例；`path`（Path）需符合函数签名约定。
        输出：返回 `None`；初始化 `MetaCheckpointStore` 的实例状态，构造参数非法时可能抛出异常。"""
        self.path = Path(path)

    def load(
        self, *, config_fingerprint: str, data_fingerprint: str, skillbank_snapshot_id: str
    ) -> dict:
        """函数作用：从配置的存储位置读取并标准化当前对象负责的数据。
        输入要求：`self` 应为已初始化的 `MetaCheckpointStore` 实例；`config_fingerprint`（str）需以关键字传入并符合签名约定；`data_fingerprint`（str）需以关键字传入并符合签名约定；`skillbank_snapshot_id`（str）需以关键字传入并符合签名约定。
        输出：返回 `dict` 类型结果；校验或下游调用失败时异常向上传递。"""
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        expected = {
            "config_fingerprint": config_fingerprint,
            "data_fingerprint": data_fingerprint,
            "skillbank_snapshot_id": skillbank_snapshot_id,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"checkpoint {key} mismatch")
        return payload

    def save(self, payload) -> None:
        """函数作用：将当前对象负责的数据安全写入持久化存储。
        输入要求：`self` 应为已初始化的 `MetaCheckpointStore` 实例；`payload`（未显式标注）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责修改对象状态或持久化文件。"""
        value = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".meta-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    value,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    default=lambda item: item.value if hasattr(item, "value") else str(item),
                )
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
