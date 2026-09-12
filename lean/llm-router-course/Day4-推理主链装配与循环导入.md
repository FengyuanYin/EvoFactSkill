# Day 4：推理主链、装配与循环导入

## 学习目标

- 理解 `InferenceRuntime` 如何统一调用两种 Router。
- 追踪路由 usage 如何进入最终 trace。
- 理解 `ExperimentRunner` 的 strategy 分支。
- 看懂 runtime 包的延迟导入。

## 一、推理主链变化

`src/evofact/runtime/inference.py::InferenceRuntime.infer` 从同步的 `router.select` 改为：

```python
routing_result = await self.router.route(public, skills, utilities, budget)
```

随后检查 `routing_result.value` 必须是 `RoutingDecision`。这是必要的运行时边界，因为 `BackendResult.value` 的静态类型是 `Any`。

usage 现在从路由阶段开始累计：

```text
router usage + specialist usage × N + judge usage = InferenceTrace.usage
```

专家仍在 for 循环里逐个 await，所以并没有因为使用 async 就自动并行。

## 二、ExperimentRunner 装配

`src/evofact/experiments/runner.py::ExperimentRunner.run` 只创建一次 backend。若 `strategy == "llm"`，它创建 `LLMSkillRouter(backend, fallback=SkillRouter(...))`；否则创建普通 `SkillRouter`。

相同 backend 同时交给 Router 和 `InferenceRuntime`，确保路由、专家分析与 Judge 使用同一模型配置。

## 三、当前 CLI 边界

`ExperimentRunner.run` 默认参数还是 `strategy="utility-aware"`。`src/evofact/cli.py` 调用时没有传 llm strategy，因此：

```text
backend=openai-compatible ≠ 自动使用 LLM Router
```

在线回归脚本会直接构造 `LLMSkillRouter`，但正式 CLI 的默认接入仍需后续配置项或命令行参数。

## 四、循环导入

Router 需要 `runtime.backend`，而 `runtime.inference` 又需要 Router。`src/evofact/runtime/__init__.py` 使用 `TYPE_CHECKING` 和模块级 `__getattr__`，在真正访问 `InferenceRuntime` 时才延迟导入，避免包初始化期间的循环依赖。

## 五、完整调用链

```text
cli._run
  → ExperimentRunner.run
  → _backend + Router 分支
  → InferenceRuntime.infer
  → await router.route
  → backend.analyze × N
  → backend.judge
  → InferenceTrace
```

## 六、自测题与答案

1. 类型注解联合会自动选择实现吗？**不会，具体对象由 Runner 构造。**
2. backend 为真实模型时 Router 必然是 LLM 吗？**不是。**
3. `__getattr__` 何时导入 inference？**运行时访问该包属性时。**
4. route usage 在哪里合并？**`InferenceRuntime.infer` 取得 routing_result 后。**

## 七、建议阅读顺序

`ExperimentRunner._backend` → `ExperimentRunner.run` → `InferenceRuntime.__init__` → `infer` → `runtime.__getattr__`。
