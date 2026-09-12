# Day 3：LLMSkillRouter 候选、校验与回退

## 学习目标

- 掌握 `LLMSkillRouter.route` 的四个阶段。
- 理解候选白名单、JSON 校验和 fallback。
- 能识别“输出合理但实际已经回退”的情况。

## 一、统一规则 Router

`SkillRouter.select` 保留原同步算法。新增 `SkillRouter.route` 只是异步适配器，把 decision 包装成 `BackendResult`，让 `InferenceRuntime` 无需按 Router 类型分支。

## 二、LLM 路由四阶段

1. 仅保留 active/frozen、作用域匹配的 `SPECIALIST`。
2. 查找 active/frozen、作用域匹配的 `ROUTER` Skill。
3. 调用 `backend.route(sample, candidates, router_skill, utilities, budget)`。
4. `_parse_decision` 将原始 JSON 转换为正式 `RoutingDecision`。

本次修复了一个关键参数错误：在线后端应收到单个 `router_skill`，而不是 `router_skills` 列表。Mock 不读取该参数，曾经掩盖这个错误；在线后端访问 `.instructions` 时则会失败。

## 三、解析器检查什么

- 响应必须是 dict；
- `selected_skill_ids` 必须是非空字符串列表；
- 去重后不能超过 `budget.max_skills`；
- ID 必须全部来自 candidates；
- confidence 必须是 `[0, 1]` 中的数字；
- reasons 非法时使用默认原因。

候选在模型调用前由程序过滤，输出后又按 ID 白名单验证，形成前后两层约束。

## 四、失败回退

网络、JSON、未知 ID、空选择、超预算和 confidence 错误都会进入 `_fallback_decision`。它调用 `SkillRouter.select`，把原因追加到 reasons，并设置 `fallback_used=True`。

```text
模型或校验成功 → fallback_used=False
任意异常       → SkillRouter.select → fallback_used=True
```

所以判断在线 Router 是否成功时，不能只看 selected，还必须检查 fallback。

## 五、自测题与答案

1. 为什么模型看不到越域 Skill？**候选在调用前已由 `matches_scope` 过滤。**
2. 重复 ID 如何处理？**`dict.fromkeys` 保序去重。**
3. 返回 4 个 ID 而预算为 3 会怎样？**校验失败并规则回退。**
4. 合理 selected 加 `fallback=True` 表示什么？**结果来自规则，而非 LLM 成功输出。**

## 六、建议阅读顺序

`SkillRouter.route` → `LLMSkillRouter.route` → `_parse_decision` → `_fallback_decision` → `matches_scope`。
