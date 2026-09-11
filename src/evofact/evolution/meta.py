import json
from pathlib import Path


class MetaEvolutionLoop:
    def __init__(self, enabled: bool = False, event_store: Path | None = None):
        """函数作用：创建并初始化 `MetaEvolutionLoop` 对象，为后续方法调用准备依赖和初始状态。
        输入要求：`self` 应为已初始化的 `MetaEvolutionLoop` 实例；`enabled`（bool，默认 `False`）需符合函数签名约定；`event_store`（Path | None，默认 `None`）需符合函数签名约定。
        输出：返回 `None`；初始化 `MetaEvolutionLoop` 的实例状态，构造参数非法时可能抛出异常。"""
        self.enabled = enabled
        self.events = []
        self.event_store = event_store

    def run(self, evolution_events: list[dict]) -> dict:
        """函数作用：执行当前对象负责的主运行流程，并汇总本轮结果。
        输入要求：`self` 应为已初始化的 `MetaEvolutionLoop` 实例；`evolution_events`（list[dict]）需符合函数签名约定。
        输出：返回 `dict` 类型结果；校验或下游调用失败时异常向上传递。"""
        if not self.enabled:
            return {"enabled": False, "proposals": []}
        record = {
            "schema_version": "meta_event_v1",
            "enabled": True,
            "source_events": len(evolution_events),
            "proposals": [],
        }
        self.events.append(record)
        if self.event_store:
            self.event_store.parent.mkdir(parents=True, exist_ok=True)
            with self.event_store.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        return record

    async def run_demse(self, runner, samples, **kwargs):
        """函数作用：负责`MetaEvolutionLoop` 中的 `run_demse` 处理，封装调用方需要复用的业务步骤。
        输入要求：`self` 应为已初始化的 `MetaEvolutionLoop` 实例；`runner`（未显式标注）需符合函数签名约定；`samples`（未显式标注）需符合函数签名约定；额外关键字参数 `**kwargs` 需为当前接口支持的选项。
        输出：异步返回函数计算得到的结果对象；具体结构由当前实现及调用方协议约定。"""
        if not self.enabled:
            raise ValueError("DEMSE meta evolution is disabled")
        outcome = await runner.run(samples, **kwargs)
        self.events.append(
            {
                "schema_version": "meta_event_v2",
                "enabled": True,
                "run_id": outcome.run_id,
                "episodes": len(outcome.episodes),
                "decisions": len(outcome.decisions),
            }
        )
        return outcome
