"""
根据一个 Skill 演化提案的实际内容生成稳定指纹，用来判断两个候选提案是否本质相同。
"""

from __future__ import annotations

import hashlib
import json

from evofact.core.models import CandidateIdentity, EvolutionProposal


def _normalize(text: str) -> str:
    """函数作用：负责当前模块中的 `_normalize` 处理，封装调用方需要复用的业务步骤。
    输入要求：`text`（str）需符合函数签名约定。
    输出：返回 `str` 类型结果；校验或下游调用失败时异常向上传递。"""
    return " ".join(text.casefold().split())


def candidate_identity(proposal: EvolutionProposal) -> CandidateIdentity:
    """函数作用：负责当前模块中的 `candidate_identity` 处理，封装调用方需要复用的业务步骤。
    输入要求：`proposal`（EvolutionProposal）需符合函数签名约定。
    输出：返回 `CandidateIdentity` 类型结果；校验或下游调用失败时异常向上传递。"""
    content = []
    parents = set()
    scopes = []
    for skill in proposal.candidate_skills:
        content.append(
            {
                "name": _normalize(skill.name),
                "kind": skill.kind.value,
                "instructions": _normalize(skill.instructions),
                "resources": sorted((k, _normalize(v)) for k, v in skill.resources.items()),
            }
        )
        parents.update(skill.parent_ids)
        scopes.append(
            {
                "domains": sorted(skill.scope.domains),
                "datasets": sorted(skill.scope.datasets),
                "windows": sorted(skill.scope.temporal_windows),
                "tags": sorted(skill.scope.tags),
            }
        )
    content_hash = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    scope_signature = hashlib.sha256(
        json.dumps(scopes, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    payload = (proposal.operation.value, sorted(parents), content_hash, scope_signature)
    fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
    return CandidateIdentity(
        fingerprint, proposal.operation, tuple(sorted(parents)), content_hash, scope_signature
    )
