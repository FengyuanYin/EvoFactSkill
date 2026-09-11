"""
轻量级的 Skill 资源安全扫描器
"""

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath

BLOCKED_IMPORTS = {"subprocess", "socket", "requests", "urllib", "shutil", "ctypes", "winreg"}
BLOCKED_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "system",
    "popen",
    "remove",
    "unlink",
    "rmtree",
}


@dataclass(frozen=True)
class SafetyReport:
    level: str
    findings: tuple[str, ...]


def scan_resources(resources: dict[str, str]) -> SafetyReport:
    """函数作用：扫描并评估 `scan_resources` 所表示的数据，供当前模块后续流程使用。
    输入要求：`resources`（dict[str, str]）需符合函数签名约定。
    输出：返回 `SafetyReport` 类型结果；校验或下游调用失败时异常向上传递。"""
    findings = []
    has_scripts = False
    for rel, source in resources.items():
        path = PurePosixPath(rel.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            findings.append(f"unsafe path: {rel}")
            continue
        if not rel.startswith("scripts/"):
            continue
        has_scripts = True
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            findings.append(f"syntax error in {rel}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [x.name.split(".")[0] for x in node.names]
                    if isinstance(node, ast.Import)
                    else [(node.module or "").split(".")[0]]
                )
                findings += [
                    f"blocked import {name} in {rel}" for name in names if name in BLOCKED_IMPORTS
                ]
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                )
                if name in BLOCKED_CALLS:
                    findings.append(f"blocked call {name} in {rel}")
    return SafetyReport(
        "blocked" if findings else ("review_required" if has_scripts else "safe"), tuple(findings)
    )
