# Day 2：统一路由协议与两种后端

## 学习目标

- 理解 `ModelBackend.route` 的输入输出。
- 区分 Protocol、Mock 和真实 HTTP 实现。
- 追踪 Router 提示词如何进入 system message。

## 一、统一协议

`src/evofact/runtime/backend.py::ModelBackend.route` 接收五类输入：公开 news、合法候选 Skill、Router Skill、历史 utility 和 `RunBudget`，异步返回 `BackendResult`。

`ModelBackend` 是 `Protocol`，不是实际执行请求的类。具体对象只要实现兼容的 `route/analyze/judge` 方法，就能注入运行时。

`BackendResult` 同时保存业务值和 `UsageRecord`。LLM 后端最初返回原始 dict，`LLMSkillRouter` 校验后将其转换成 `RoutingDecision`。

## 二、MockBackend.route

`src/evofact/runtime/mock_backend.py::MockBackend.route` 用关键词选择 Skill，例如数字对应 `numerical_consistency`，时间词对应 `temporal_reasoning`。

它明确丢弃 `router_skill` 和 utilities，因此只适合测试接口与控制流，不能评价提示词质量。本次还修正了 `reason`/`reasons` 字段名和时间 Skill 名称。

## 三、OpenAICompatibleBackend.route

`src/evofact/runtime/openai_backend.py::OpenAICompatibleBackend.route` 将候选转成 JSON 视图，包含 ID、名称、instructions、scope、triggers 和 utility。

system message 的来源是：

```text
skills/seeds/coordinator_routing/SKILL.md
  → load_skill_package
  → SkillSpec.instructions
  → router_skill.instructions + 固定安全/JSON约束
```

用户消息包含 sample、budget 和 candidates。`_call` 发送 `/chat/completions` 请求，并设置 `response_format=json_object` 与 `temperature=0`，减少回归测试波动。

## 四、完整调用链

```text
LLMSkillRouter.route
  → OpenAICompatibleBackend.route
  → OpenAICompatibleBackend._call
  → HTTP API
  → JSON dict
  → LLMSkillRouter._parse_decision
```

`tests/test_router_prompt.py::test_router_prompt_and_candidates_reach_backend` 用捕获后端验证 instructions、sample 和 candidates 确实进入请求，全程不访问网络。

## 五、自测题与答案

1. Mock 通过能证明提示词有效吗？**不能。**
2. 谁真正发送 HTTP？**`OpenAICompatibleBackend._call` 内的 send。**
3. 提示词位于哪个消息角色？**system。**
4. 当前 route usage 是否包含真实 token？**没有，目前主要记录 calls。**

## 六、建议阅读顺序

`BackendResult` → `ModelBackend` → `MockBackend.route` → `OpenAICompatibleBackend.route` → `_call`。
