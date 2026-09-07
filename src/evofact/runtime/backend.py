from dataclasses import dataclass
from typing import Any, Protocol
from evofact.core.models import SkillSpec, SpecialistReport, Prediction, UsageRecord

@dataclass(frozen=True)
class BackendResult:
    value: Any
    usage: UsageRecord=UsageRecord()

class ModelBackend(Protocol):
    async def analyze(self,sample:dict[str,Any],skill:SkillSpec)->BackendResult: ...
    async def judge(self,sample:dict[str,Any],reports:tuple[SpecialistReport,...],skill:SkillSpec)->BackendResult: ...
