# Day 1：未提交改动全景与学习地图

## 学习目标

- 理解这批改动为何把规则路由扩展为 LLM 路由。
- 区分 Router 选择现有 Skill 与 evolution 创建新 Skill。
- 找到协议、实现、装配和测试四层代码。

## 一、改动目标

原来的 `src/evofact/routing/router.py::SkillRouter.select` 依靠作用域、触发词、历史效用和固定策略选择专家。新增路径允许 LLM 阅读 news、Router 提示词与候选 Skill 描述，再输出候选 ID。

```text
规则路径：news → SkillRouter → RoutingDecision
LLM 路径：news → LLMSkillRouter → backend.route → JSON 校验 → RoutingDecision
```

两条路径最终返回相同的数据结构，因此专家分析和 Judge 不需要感知 Router 类型。

## 二、改动文件地图

| 层次 | 文件 | 责任 |
|---|---|---|
| 路由 | `src/evofact/routing/router.py` | LLM 路由、校验与规则回退 |
| 协议 | `src/evofact/runtime/backend.py` | 声明 `ModelBackend.route` |
| 后端 | `mock_backend.py`、`openai_backend.py` | 离线和真实模型实现 |
| 推理 | `src/evofact/runtime/inference.py` | 异步调用 Router 并累计 usage |
| 装配 | `src/evofact/experiments/runner.py` | 根据 strategy 创建 Router |
| 测试 | `tests/test_router_prompt.py` | 离线验证提示词传递 |
| 回归 | `tests/router_prompt_regression.py` | 用真实 LLM 评价语义选择 |

这里的顺序是根据当前 `git diff` 重建的依赖顺序，并非 commit 历史，因为改动尚未提交。

## 三、责任边界

Router 只从已有 `SPECIALIST` Skill 中选择候选，不创建新的 `SkillSpec`，也不判断新闻最终是 `REAL` 或 `FAKE`。新 Skill 由 evolution/proposer 流程生成，最终真假由 Judge 决定。

## 四、当前完成度

已经完成：统一异步接口、LLM 候选选择、白名单校验、规则回退、推理接入、离线测试和手动在线回归。

仍需注意：`ExperimentRunner.run` 默认 strategy 还是 `utility-aware`，普通 CLI 没有传 `strategy="llm"`，所以主 CLI 尚未默认启用 LLM Router。

## 五、自测题与答案

1. Router 会创建新 Skill 吗？**不会，只选择已有专家。**
2. 为什么保留规则 Router？**支持原策略，并作为 LLM 失败后的确定性回退。**
3. 在线后端与 LLM Router 是同一个对象吗？**不是，Router 管控制流程，backend 负责模型调用。**
4. CLI 当前默认使用什么？**`utility-aware` 规则路由。**

## 六、建议阅读顺序

`backend.py` → `router.py` → `openai_backend.py` → `inference.py` → `runner.py` → 两个新增测试文件。
