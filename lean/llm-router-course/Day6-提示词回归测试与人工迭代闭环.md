# Day 6：提示词回归测试与人工迭代闭环

## 学习目标

- 区分离线工程测试与在线语义回归。
- 理解 6 个 RouterCase 的验收标准。
- 能按单个失败案例迭代提示词。
- 正确解释 PASS、FAIL、fallback 和退出码。

## 一、两层测试

`tests/test_router_prompt.py` 使用 `CapturingBackend`，不访问网络，验证 Router instructions、news sample 和全部 specialist candidates 到达 OpenAI 后端边界。

`tests/router_prompt_regression.py` 只有添加 `--live` 才访问 API，负责判断真实模型的 Skill 选择是否符合人工期望。

## 二、回归案例

| case_id | 期望 Skill |
|---|---|
| `numerical_mismatch` | `numerical_consistency` |
| `future_evidence` | `temporal_reasoning` |
| `conflicting_sources` | `cross_source_contradiction` |
| `anonymous_source` | `source_credibility` |
| `sensational_framing` | `linguistic_manipulation` |
| `number_and_timeline` | numerical + temporal |

每个案例的 `max_skills` 等于期望数量，因此要求实际集合与期望集合完全一致、没有回退且不超预算。

## 三、执行方法

查看案例，不调用 API：

```powershell
python tests/router_prompt_regression.py
```

设置当前会话的密钥并运行全部案例：

```powershell
$env:DEEPSEEK_API_KEY = "你的 API Key"
python tests/router_prompt_regression.py --live
```

只运行一个案例：

```powershell
python tests/router_prompt_regression.py --live --case numerical_mismatch
```

普通离线测试：

```powershell
python -m pytest -q
```

## 四、读懂失败

`fallback=False` 但 actual 与 expected 不同，表示调用链正常、提示词语义选择错误。

`fallback=True` 表示 API、响应格式或结果校验失败；actual 来自规则回退，此时应先排查工程问题。

脚本打印当前 `coordinator_routing/SKILL.md` 的 SHA256 摘要。修改后摘要变化，证明新提示词被重新加载。全部通过返回 0，有语义失败返回 1，测试配置错误返回 2。

## 五、人工迭代闭环

```text
记录失败案例
  → 修改 skills/seeds/coordinator_routing/SKILL.md
  → 单独运行失败案例
  → 通过后运行全部案例
  → 检查旧案例是否退化
```

应描述通用决策边界，不要把 case ID 写进提示词，否则是在记忆测试而不是理解 news。

## 六、测试边界

测试只能说明当前模型、配置和固定样本下，从已有 Skill 中选择正确；不能证明所有新闻都正确、不能创建新 Skill，也不能证明普通 CLI 已默认启用 LLM Router。

## 七、自测题与答案

1. 为什么在线脚本不作为普通 pytest 自动运行？**避免自动产生网络调用和费用。**
2. fallback=True 时先改提示词吗？**不，应先排查连接、JSON 和校验。**
3. 为什么限制 max_skills？**验证最小充分候选集合。**
4. 全部案例通过能证明 CLI 已接入吗？**不能，CLI 当前未传 llm strategy。**

## 八、建议阅读顺序

`RouterCase` → `ROUTER_CASES` → `run_live` → `main` → `CapturingBackend` → 离线测试函数。
