import asyncio
import json
import urllib.request

from evofact.core.models import Prediction, SkillSpec, SpecialistReport, UsageRecord

from .backend import BackendResult


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, api_key: str, model: str):
        """函数作用：创建并初始化 `OpenAICompatibleBackend` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `OpenAICompatibleBackend` 实例；`base_url`（str）需符合函数签名约定；`api_key`（str）需符合函数签名约定；`model`（str）需符合函数签名约定。
        输出：返回 `None`；初始化 `OpenAICompatibleBackend` 的实例状态，构造参数非法时可能抛出异常。"""
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = api_key
        self.model = model

    async def _call(self, system: str, payload: dict) -> dict:
        """函数作用：负责`OpenAICompatibleBackend` 中的 `_call` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `OpenAICompatibleBackend` 实例；`system`（str）需符合函数签名约定；`payload`（dict）需符合函数签名约定。
        输出：异步返回 `dict` 类型结果；校验或下游调用失败时异常向上传递。"""
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
            """函数作用：负责当前模块中的 `send` 处理，封装调用方需要复用的业务步骤。
            输入要求：无显式输入；若函数位于另一函数内部，则依赖已初始化的外层变量。
            输出：返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read())

        data = await asyncio.to_thread(send)
        return json.loads(data["choices"][0]["message"]["content"])

    async def analyze(self, sample: dict, skill: SkillSpec) -> BackendResult:
        """函数作用：负责`OpenAICompatibleBackend` 中的 `analyze` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `OpenAICompatibleBackend` 实例；`sample`（dict）需符合函数签名约定；`skill`（SkillSpec）需符合函数签名约定。
        输出：异步返回 `BackendResult` 类型结果；校验或下游调用失败时异常向上传递。"""
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
        """函数作用：负责`OpenAICompatibleBackend` 中的 `judge` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `OpenAICompatibleBackend` 实例；`sample`（dict）需符合函数签名约定；`reports`（tuple[SpecialistReport, ...]）需符合函数签名约定；`skill`（SkillSpec）需符合函数签名约定。
        输出：异步返回 `BackendResult` 类型结果；校验或下游调用失败时异常向上传递。"""
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
