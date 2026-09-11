from __future__ import annotations

import difflib
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from evofact.core.models import SkillKind, SkillScope, SkillSpec, SkillStatus, Trigger


def _jsonable(value):
    """函数作用：负责当前模块中的 `_jsonable` 处理，封装调用方需要复用的业务步骤。
    输入要求：`value`（未显式标注）需符合函数签名约定。
    输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _skill_from(data: dict) -> SkillSpec:
    """函数作用：负责当前模块中的 `_skill_from` 处理，封装调用方需要复用的业务步骤。
    输入要求：`data`（dict）需符合函数签名约定。
    输出：返回 `SkillSpec` 类型结果；校验或下游调用失败时异常向上传递。"""
    data = dict(data)
    data["kind"] = SkillKind(data["kind"])
    data["status"] = SkillStatus(data["status"])
    data["scope"] = SkillScope(**{k: tuple(v) for k, v in data.get("scope", {}).items()})
    data["triggers"] = tuple(Trigger(**x) for x in data.get("triggers", []))
    data["parent_ids"] = tuple(data.get("parent_ids", []))
    return SkillSpec(**data)


class SkillRepository:
    def __init__(self, root: Path):
        """函数作用：创建并初始化 `SkillRepository` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`root`（Path）需符合函数签名约定。
        输出：返回 `None`；初始化 `SkillRepository` 的实例状态，构造参数非法时可能抛出异常。"""
        self.root = Path(root)
        self.snapshots = self.root / "snapshots"
        self.events = self.root / "events.jsonl"
        self.active_file = self.root / "active.json"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def _atomic_json(self, path: Path, value) -> None:
        """函数作用：负责`SkillRepository` 中的 `_atomic_json` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`path`（Path）需符合函数签名约定；`value`（未显式标注）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责修改对象状态或持久化文件。"""
        if path == self.active_file and "schema_version" not in value and path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("schema_version") == "active_v2":
                value = {**previous, "skills": value}
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(_jsonable(value), stream, ensure_ascii=False, sort_keys=True, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _event(self, action: str, **payload) -> None:
        """函数作用：负责`SkillRepository` 中的 `_event` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`action`（str）需符合函数签名约定；额外关键字参数 `**payload` 需为当前接口支持的选项。
        输出：返回 `None`；可能按函数职责修改对象状态或持久化文件。"""
        event = {
            "schema_version": "skill_event_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **_jsonable(payload),
        }
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def _active(self) -> dict[str, str]:
        """函数作用：负责`SkillRepository` 中的 `_active` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；无其他显式输入。
        输出：返回 `dict[str, str]` 类型结果；校验或下游调用失败时异常向上传递。"""
        payload = (
            json.loads(self.active_file.read_text(encoding="utf-8"))
            if self.active_file.exists()
            else {}
        )
        return (
            payload.get("skills", payload)
            if payload.get("schema_version") == "active_v2"
            else payload
        )

    def transaction(self, run_id: str) -> dict | None:
        """函数作用：负责`SkillRepository` 中的 `transaction` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`run_id`（str）需符合函数签名约定。
        输出：返回 `dict | None` 类型结果；校验或下游调用失败时异常向上传递。"""
        if not self.active_file.exists():
            return None
        payload = json.loads(self.active_file.read_text(encoding="utf-8"))
        return (
            payload.get("transactions", {}).get(run_id)
            if payload.get("schema_version") == "active_v2"
            else None
        )

    def commit_bank(
        self, skills: list[SkillSpec], *, run_id: str, baseline: list[SkillSpec], audit: dict
    ) -> tuple[str, ...]:
        """函数作用：通过一次原子指针写入提交完整技能库及其审计记录。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`skills`（list[SkillSpec]）需符合函数签名约定；`run_id`（str）需以关键字传入并符合签名约定；`baseline`（list[SkillSpec]）需以关键字传入并符合签名约定；`audit`（dict）需以关键字传入并符合签名约定。
        输出：返回 `tuple[str, ...]` 类型结果；校验或下游调用失败时异常向上传递。"""
        from evofact.data.domains import skillbank_fingerprint

        previous = self.transaction(run_id)
        if previous:
            if skillbank_fingerprint(list(self.active().values())) != previous["after"]:
                raise ValueError("repository changed after transaction")
            return tuple(previous["snapshots"])
        current = list(self.active().values())
        if current and skillbank_fingerprint(current) != skillbank_fingerprint(baseline):
            raise ValueError("repository changed during evaluation")
        if len({s.name for s in skills}) != len(skills):
            raise ValueError("duplicate names in atomic bank")
        active = {}
        for skill in skills:
            data = _jsonable(asdict(skill))
            digest = hashlib.sha256(
                json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            path = self.snapshots / f"{digest}.json"
            if not path.exists():
                self._atomic_json(path, data)
            active[skill.name] = digest
        payload = (
            json.loads(self.active_file.read_text(encoding="utf-8"))
            if self.active_file.exists()
            else {}
        )
        transactions = (
            payload.get("transactions", {}) if payload.get("schema_version") == "active_v2" else {}
        )
        transactions[run_id] = {
            "before": skillbank_fingerprint(baseline),
            "after": skillbank_fingerprint(skills),
            "snapshots": list(active.values()),
            "audit": audit,
        }
        self._atomic_json(
            self.active_file,
            {"schema_version": "active_v2", "skills": active, "transactions": transactions},
        )
        return tuple(active.values())

    def diff(self, name: str, snapshot: str | None = None) -> str:
        """函数作用：负责`SkillRepository` 中的 `diff` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`name`（str）需符合函数签名约定；`snapshot`（str | None，默认 `None`）需符合函数签名约定。
        输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
        current = self._active()[name]
        if snapshot is None:
            choices = [
                event["snapshot"]
                for event in self.history()
                if event.get("name") == name and event.get("snapshot") != current
            ]
            snapshot = choices[-1] if choices else current
        old = self.get_snapshot(snapshot)
        if old.name != name:
            raise ValueError("diff snapshot belongs to another skill")
        left = json.dumps(
            _jsonable(asdict(old)), ensure_ascii=False, sort_keys=True, indent=2
        ).splitlines(True)
        right = json.dumps(
            _jsonable(asdict(self.get_snapshot(current))),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).splitlines(True)
        return "".join(difflib.unified_diff(left, right, fromfile=snapshot, tofile=current))

    def save(self, skill: SkillSpec, action: str = "stage") -> str:
        """函数作用：将当前对象负责的数据安全写入持久化存储。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`skill`（SkillSpec）需符合函数签名约定；`action`（str，默认 `'stage'`）需符合函数签名约定。
        输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
        data = _jsonable(asdict(skill))
        digest = hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        path = self.snapshots / f"{digest}.json"
        if not path.exists():
            self._atomic_json(path, data)
        self._event(
            action, skill_id=skill.skill_id, snapshot=digest, name=skill.name, status=skill.status
        )
        return digest

    def get_snapshot(self, digest: str) -> SkillSpec:
        """函数作用：负责`SkillRepository` 中的 `get_snapshot` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`digest`（str）需符合函数签名约定。
        输出：返回 `SkillSpec` 类型结果；校验或下游调用失败时异常向上传递。"""
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("snapshot must be a SHA-256 digest")
        return _skill_from(
            json.loads((self.snapshots / f"{digest}.json").read_text(encoding="utf-8"))
        )

    def list(self, status: SkillStatus | None = None) -> list[SkillSpec]:
        """函数作用：负责`SkillRepository` 中的 `list` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`status`（SkillStatus | None，默认 `None`）需符合函数签名约定。
        输出：返回 `list[SkillSpec]` 类型结果；校验或下游调用失败时异常向上传递。"""
        result = [self.get_snapshot(p.stem) for p in sorted(self.snapshots.glob("*.json"))]
        return [s for s in result if status is None or s.status == status]

    def active(self) -> dict[str, SkillSpec]:
        """函数作用：负责`SkillRepository` 中的 `active` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；无其他显式输入。
        输出：返回 `dict[str, SkillSpec]` 类型结果；校验或下游调用失败时异常向上传递。"""
        return {n: self.get_snapshot(d) for n, d in self._active().items()}

    def promote(self, skill: SkillSpec) -> str:
        """函数作用：负责`SkillRepository` 中的 `promote` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`skill`（SkillSpec）需符合函数签名约定。
        输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
        if skill.status == SkillStatus.RETIRED:
            raise ValueError("cannot promote retired skill")
        promoted = replace(skill, status=SkillStatus.ACTIVE)
        digest = self.save(promoted, "promote")
        active = self._active()
        active[promoted.name] = digest
        self._atomic_json(self.active_file, active)
        return digest

    def promote_batch(self, skills: list[SkillSpec]) -> list[str]:
        """函数作用：负责`SkillRepository` 中的 `promote_batch` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`skills`（list[SkillSpec]）需符合函数签名约定。
        输出：返回 `list[str]` 类型结果；校验或下游调用失败时异常向上传递。"""
        promoted = []
        digests = []
        for skill in skills:
            if skill.status == SkillStatus.RETIRED:
                raise ValueError("cannot promote retired skill")
            item = replace(skill, status=SkillStatus.ACTIVE)
            data = _jsonable(asdict(item))
            digest = hashlib.sha256(
                json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            promoted.append(item)
            digests.append(digest)
        for item, digest in zip(promoted, digests):
            path = self.snapshots / f"{digest}.json"
            if not path.exists():
                self._atomic_json(path, _jsonable(asdict(item)))
        active = self._active()
        for item, digest in zip(promoted, digests):
            active[item.name] = digest
        self._atomic_json(self.active_file, active)
        for item, digest in zip(promoted, digests):
            self._event(
                "meta_promote",
                skill_id=item.skill_id,
                snapshot=digest,
                name=item.name,
                status=item.status,
            )
        return digests

    def freeze(self, name: str) -> str:
        """函数作用：负责`SkillRepository` 中的 `freeze` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`name`（str）需符合函数签名约定。
        输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
        skill = self.active()[name]
        frozen = replace(skill, status=SkillStatus.FROZEN)
        digest = self.save(frozen, "freeze")
        active = self._active()
        active[name] = digest
        self._atomic_json(self.active_file, active)
        return digest

    def retire(self, name: str, reason: str) -> str:
        """函数作用：负责`SkillRepository` 中的 `retire` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`name`（str）需符合函数签名约定；`reason`（str）需符合函数签名约定。
        输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
        skill = self.active()[name]
        retired = replace(skill, status=SkillStatus.RETIRED)
        digest = self.save(retired, "retire")
        active = self._active()
        active.pop(name, None)
        self._atomic_json(self.active_file, active)
        self._event("retire_reason", name=name, reason=reason)
        return digest

    def rollback(self, name: str, digest: str) -> None:
        """函数作用：负责`SkillRepository` 中的 `rollback` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；`name`（str）需符合函数签名约定；`digest`（str）需符合函数签名约定。
        输出：返回 `None`；可能按函数职责修改对象状态或持久化文件。"""
        skill = self.get_snapshot(digest)
        if skill.name != name or skill.status == SkillStatus.RETIRED:
            raise ValueError("invalid rollback target")
        active = self._active()
        active[name] = digest
        self._atomic_json(self.active_file, active)
        self._event("rollback", name=name, snapshot=digest)

    def history(self) -> list[dict]:
        """函数作用：负责`SkillRepository` 中的 `history` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `SkillRepository` 实例；无其他显式输入。
        输出：返回 `list[dict]` 类型结果；校验或下游调用失败时异常向上传递。"""
        events = (
            [json.loads(x) for x in self.events.read_text(encoding="utf-8").splitlines()]
            if self.events.exists()
            else []
        )
        payload = (
            json.loads(self.active_file.read_text(encoding="utf-8"))
            if self.active_file.exists()
            else {}
        )
        if payload.get("schema_version") == "active_v2":
            for run_id, transaction in payload.get("transactions", {}).items():
                for digest in transaction["snapshots"]:
                    skill = self.get_snapshot(digest)
                    events.append(
                        {
                            "action": "meta_promote",
                            "run_id": run_id,
                            "snapshot": digest,
                            "name": skill.name,
                            "skill_id": skill.skill_id,
                            "status": skill.status,
                            "audit": transaction["audit"],
                        }
                    )
        return events
