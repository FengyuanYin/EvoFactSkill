import asyncio
import json
import urllib.request

from evofact.core.models import Prediction, SkillSpec, SpecialistReport, UsageRecord

from .backend import BackendResult


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = api_key
        self.model = model

    async def _call(self, system: str, payload: dict) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, default=str),
                    },
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode()
        req = urllib.request.Request(
            self.url,
            body,
            {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )

        def send():
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read())

        data = await asyncio.to_thread(send)
        return json.loads(data["choices"][0]["message"]["content"])

    async def analyze(self, sample: dict, skill: SkillSpec) -> BackendResult:
        data = await self._call(skill.instructions, sample)
        report = SpecialistReport(
            skill.skill_id,
            tuple(data.get("claims", [])),
            (),
            str(data.get("assessment", "unknown")),
            float(data.get("confidence", 0)),
            tuple(data.get("limitations", [])),
        )
        return BackendResult(report, UsageRecord(calls=1))

    async def judge(
        self, sample: dict, reports: tuple[SpecialistReport, ...], skill: SkillSpec
    ) -> BackendResult:
        data = await self._call(
            skill.instructions, {"sample": sample, "reports": [r.model_dump() for r in reports]}
        )
        return BackendResult(
            Prediction(
                str(data.get("label", "ABSTAIN")).upper(),
                float(data.get("confidence", 0)),
                str(data.get("rationale", "")),
            ),
            UsageRecord(calls=1),
        )
