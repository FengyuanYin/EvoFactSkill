from __future__ import annotations

import hashlib
import json

from evofact.core.models import CandidateIdentity, EvolutionProposal


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def candidate_identity(proposal: EvolutionProposal) -> CandidateIdentity:
    content = []
    parents = set()
    scopes = []
    for skill in proposal.candidate_skills:
        content.append({"name": _normalize(skill.name), "kind": skill.kind.value, "instructions": _normalize(skill.instructions), "resources": sorted((k, _normalize(v)) for k, v in skill.resources.items())})
        parents.update(skill.parent_ids)
        scopes.append({"domains": sorted(skill.scope.domains), "datasets": sorted(skill.scope.datasets), "windows": sorted(skill.scope.temporal_windows), "tags": sorted(skill.scope.tags)})
    content_hash = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    scope_signature = hashlib.sha256(json.dumps(scopes, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload = (proposal.operation.value, sorted(parents), content_hash, scope_signature)
    fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
    return CandidateIdentity(fingerprint, proposal.operation, tuple(sorted(parents)), content_hash, scope_signature)
