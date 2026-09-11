"""
根据一组具有相同错误模式的归因报告，自动生成一个 Skill 演化提案。
它会判断：
- 是否能找到主要责任 Skill；
- 如果能找到，就生成一个修改现有 Skill 的 EDIT 提案；
- 如果找不到，就生成一个新增 Skill 的 ADD 提案。
这同样是基于固定模板的规则生成，不会调用 LLM。
"""

import hashlib
from dataclasses import replace

from evofact.core.models import (
    AttributionReport,
    EvolutionOperation,
    EvolutionProposal,
    SkillKind,
    SkillSpec,
    SkillStatus,
)


def propose_from_cluster(
    cluster_id: str, reports: list[AttributionReport], bank: list[SkillSpec]
) -> EvolutionProposal:
    """函数作用：负责当前模块中的 `propose_from_cluster` 处理，封装调用方需要复用的业务步骤。
    输入要求：`cluster_id`（str）需符合函数签名约定；`reports`（list[AttributionReport]）需符合函数签名约定；`bank`（list[SkillSpec]）需符合函数签名约定。
    输出：返回 `EvolutionProposal` 类型结果；校验或下游调用失败时异常向上传递。"""
    responsible = [s for r in reports for s in r.responsible_skill_ids]
    target = (
        next(
            (
                sid
                for sid in responsible
                if responsible.count(sid) == max(map(responsible.count, responsible))
            ),
            None,
        )
        if responsible
        else None
    )
    trace_ids = tuple(r.trace_id for r in reports)
    error = next((e.value for r in reports for e in r.error_types), "unclassified")
    if target and any(s.skill_id == target for s in bank):
        old = next(s for s in bank if s.skill_id == target)
        candidate = replace(
            old,
            skill_id=hashlib.sha256(f"{old.skill_id}:{cluster_id}".encode()).hexdigest()[:20],
            version=_bump(old.version),
            status=SkillStatus.CANDIDATE,
            instructions=old.instructions
            + f"\nHandle recurring {error} cases by explicitly checking the missing condition.",
            parent_ids=(old.skill_id,),
        )
        operation = EvolutionOperation.EDIT
        targets = (old.name,)
    else:
        name = f"discovered_{error}"[:63]
        candidate = SkillSpec(
            hashlib.sha256(f"{name}:{cluster_id}".encode()).hexdigest()[:20],
            name,
            SkillKind.SPECIALIST,
            "0.1.0",
            SkillStatus.CANDIDATE,
            f"Detect and mitigate {error} without using labels or dataset identity.",
        )
        operation = EvolutionOperation.ADD
        targets = ()
    return EvolutionProposal(
        f"proposal-{hashlib.sha1(cluster_id.encode()).hexdigest()[:10]}",
        operation,
        f"Recurring error cluster: {error}",
        targets,
        (candidate,),
        trace_ids,
        cluster_id,
    )


def _bump(version: str) -> str:
    """函数作用：负责当前模块中的 `_bump` 处理，封装调用方需要复用的业务步骤。
    输入要求：`version`（str）需符合函数签名约定。
    输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except ValueError:
        return version + ".1"
