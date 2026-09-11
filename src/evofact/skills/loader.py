import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from evofact.core.models import SkillKind, SkillScope, SkillSpec, SkillStatus, Trigger


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    """函数作用：负责当前模块中的 `_frontmatter` 处理，封装调用方需要复用的业务步骤。
    输入要求：`text`（str）需符合函数签名约定。
    输出：返回 `tuple[dict[str, str], str]` 类型结果；校验或下游调用失败时异常向上传递。"""
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    head, marker, body = text[4:].partition("\n---\n")
    if not marker:
        raise ValueError("unterminated frontmatter")
    meta = {}
    for line in head.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, body.strip()


def _safe_path(rel: str) -> None:
    """函数作用：负责当前模块中的 `_safe_path` 处理，封装调用方需要复用的业务步骤。
    输入要求：`rel`（str）需符合函数签名约定。
    输出：返回 `None`；可能按函数职责修改对象状态或持久化文件。"""
    path = PurePosixPath(rel.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe skill resource path: {rel}")


def load_skill_package(directory: Path, *, status: SkillStatus = SkillStatus.ACTIVE) -> SkillSpec:
    """函数作用：读取并转换 `load_skill_package` 所表示的数据，供当前模块后续流程使用。
    输入要求：`directory`（Path）需符合函数签名约定；`status`（SkillStatus，默认 `SkillStatus.ACTIVE`）需以关键字传入并符合签名约定。
    输出：返回 `SkillSpec` 类型结果；校验或下游调用失败时异常向上传递。"""
    meta, instructions = _frontmatter((directory / "SKILL.md").read_text(encoding="utf-8"))
    name = meta.get("name", "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name):
        raise ValueError(f"invalid skill name: {name}")
    extra = {}
    if (directory / "metadata.json").is_file():
        extra = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    resources = {}
    for folder in ("references", "templates", "assets", "scripts"):
        base = directory / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                rel = path.relative_to(directory).as_posix()
                _safe_path(rel)
                if path.stat().st_size > 200_000:
                    raise ValueError(f"resource too large: {rel}")
                resources[rel] = path.read_text(encoding="utf-8")
    payload = json.dumps(
        {"meta": meta, "extra": extra, "instructions": instructions, "resources": resources},
        sort_keys=True,
        ensure_ascii=False,
    )
    sid = hashlib.sha256(payload.encode()).hexdigest()[:20]
    scope = SkillScope(
        tuple(extra.get("domains", [])),
        tuple(extra.get("datasets", [])),
        tuple(extra.get("temporal_windows", [])),
        tuple(extra.get("tags", [])),
    )
    triggers = tuple(Trigger(**x) for x in extra.get("triggers", []))
    return SkillSpec(
        sid,
        name,
        SkillKind(meta.get("kind", "specialist")),
        meta.get("version", "0.1.0"),
        status,
        instructions,
        resources,
        scope,
        triggers,
        tuple(extra.get("parent_ids", [])),
        extra.get("safety_level", "text_only"),
    )
