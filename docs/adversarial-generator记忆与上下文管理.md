# Adversarial Generator 记忆与上下文管理

该功能只接入 `adversarial-evolve` 的 `AdversarialBatchRunner`，默认关闭。普通 evolve、feedback-evolve、meta-evolve 及其他历史生成工作流不自动启用；Verifier 继续盲审，Generator 的 SKILL.md 不自动修改。

## 新实验配置

```yaml
generator_memory:
  enabled: true
  context_window: null
  reserved_output_tokens: 8192
  safety_fraction: 0.1
  max_overflow_retries: 2
  tokenizer_encoding: null
```

自动窗口来自 **Generator 实际调用后端** 的模型元数据，独立于 optimizer 窗口。当前生成器使用配置中的检测后端，或测试注入的 generation backend。服务不提供窗口时，在真实训练推理前保存 `generator_context_failure` 并停止；建立新实验时显式填写服务商确认的窗口，不能凭模型名字猜测。

临时查询失败不会成为永久空缓存；服务恢复后可用原配置 `--resume` 再次查询。接口本身不提供窗口时，才需要使用确认过窗口大小的新配置。

启用后，预算覆盖生成器系统指令、路线附加规则、references、template、当前训练示例、技能上下文、已消费反馈及生成记忆。可用输入为 `floor(context_window × 0.9) − 输出预留`。指定匹配的可用 tokenizer 时按 tokenizer 计数，否则使用 UTF-8 字节数加协议余量保守估算。实际输入用量高于估算时向上校准后续计数。

## 离线准备与启动

下面只创建新的两折配置和名单，不调用模型：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_independent_validation_run.py --routes adversarial-evolve --samples-per-domain 32 --evaluation-samples-per-domain 4 --optimizer-memory --generator-memory
```

本次已生成：

```powershell
$runRoot = 'docs/independent-validation/20261019-150845-86686b3d'
$route = 'adversarial-evolve'
```

使用 [独立均衡验证启动教程](独立均衡验证实验启动教程-Windows.md) 的两折训练及测试循环，将其中 `$runRoot` 替换为上面路径。每折训练 ID、独立候选池、发现验证池不变；候选池每域 32 条，每次每域抽 4 条。

若服务不支持模型窗口查询，重新准备时追加 `--generator-context-window` 和服务商确认的窗口整数；optimizer 的窗口独立使用 `--optimizer-context-window` 指定。启用新功能改变实验身份，应使用新配置和新目录；已有配置、checkpoint、技能库与结果保留。

## 四类生成经验

| 类型 | 记忆内容 | 用途 |
| --- | --- | --- |
| 协议问题 | JSON 解析失败、数量/配对/来源/重复等结构拒绝及出现次数 | 避免重复生成不可用输出 |
| 标签分歧 | 策略、领域、目标标签、复核状态、最近复核理由、来源轮次 | 改进标签可判断性；不是强制顺从复核器改标 |
| 能力边界 | 策略、领域、标签、检测技能库摘要、完整推理后判对/判错/运行失败数量 | 对旧弱点标记版本，对当前弱点扩展变体 |
| 覆盖统计 | 通过结构检查的候选按策略、领域、目标标签计数 | 避免只生成一种模式，保留探索空间 |

结构不合法的候选不贡献“有效覆盖”；复核不通过的候选不贡献检测能力边界。运行时错误、异常或非法业务输出单独计数，不作为检测器被成功难倒。完整输出但判错才是该版本上的业务错误；判对案例也保留，不通过反复采样寻找一次错误。

能力边界绑定检测技能库摘要，并明确 `current_bank`。它只说明过去在那个版本观察到什么，不保证旧弱点在新版本上仍存在。当前版本没有重测时不会自动标记“已解决”；第一版不自动重放旧题或追加生成预算。

来源 label 契约决定合法业务标签，不硬编码只支持 REAL/FAKE。Weibo21 中对应 REAL/FAKE；训练真实标签仍是 Generator 的授权监督。复核结果标记为模型复核观察，不升级为绝对事实标签。

## 短期、长期与正文检索

本批补充生成和下一批初始生成均能读取此前已完成的核验与检测观察。统计由程序生成，不额外调用 LLM 总结。长期上下文优先放结构化统计；少量代表生成案例可以检索补充，运行失败或尚未评分的案例不作为能力代表案例。

同域经验优先，重复出现的问题保留次数，优先级相同的记录按最后一次观察轮次从新到旧排列，不按字典首次插入顺序。协议、标签、能力、覆盖统计和检索案例都记录 `last_seen`，它是本运行中的相对轮次序号。旧原始案例退出本次请求时，其审计和统计仍保留在 checkpoint，不删除历史文件。

历史案例只能帮助理解变换方法。新样本的 `source_sample_id` 和 `source_trace_ids` 必须仍指向**本次提供的训练 examples**，不能把历史生成新闻当成新的真实训练来源。此前已接收挑战参与跨批规范化文本与 ID 去重，避免换个 ID 反复提交同一新闻。

## Token 不足时的缩减顺序

1. 移出旧的检索案例。
2. 减少覆盖、能力边界、标签分歧及协议问题的低优先级统计。
3. 移出较旧的批内验证反馈；最近反馈可以移出被拒包文件，减少重复 trace。
4. 将当前检测技能包的文件内容退化为包目录信息；Generator 不负责编辑包，无需保留全部技能文件才能生成。
5. 减少本轮训练 examples，至少保留一个完整来源样本及 trace。

生成规则、输出协议、标签契约、references/template 和最后一个完整来源示例不按字符截断。仍放不下则停止保存，不静默省略标签或来源依据。旧的总字符裁剪只在功能关闭时沿用；启用后按实际模型 token 窗口组装整个请求。

仅明确的输入上下文超限触发缩减后的有限重试。超时、429、输出截断及解析失败分别审计，不触发自动记忆淘汰重试；手动恢复失败请求不消耗“上下文超限重试”次数。所有实际请求仍受全局调用、费用和 token 预算约束。

服务已经返回的 usage 会随异常跨异步任务传递；即使输出被截断或 JSON 无法解析，实际 token 和可定价费用仍结算。没有返回的用量保持未知，修复前未保存的历史 usage 不自动补账。低估校准在已有系数上继续向上调整。

## 隔离与核验

记忆位于当前运行的 checkpoint，绑定 run_id 和生成器包摘要，不自动从其他实验或 fold 导入。两折训练/测试领域互换，不能共享具体新闻记忆。

组装记忆时过滤与当前受保护样本重合的观察和案例，实际请求发送前再次执行防泄漏检查。独立均衡验证正文、标签、逐样本预测和 final-test 不进入生成记忆。

Verifier 的请求仍只有待审文本、可选 evidence、标签契约及关联源训练文本/gold 标签。Generator 的记忆、拟定目标标签、策略和理由不传给 Verifier，复核规则与门槛保持原样。

## 断点和审计

| 字段 | 内容 |
| --- | --- |
| `generator_memory` | 本运行的版本、逐轮经验、标签与检测观察 |
| `generator_requests` | 每轮实际 payload、响应、预算、淘汰项及调用/重试用量 |
| `generator_capabilities` | Generator 自己的 endpoint/model 窗口元数据 |
| `generator_token_factor` | 根据实际用量向上校准的估算系数 |
| `generator_context_error` | 当前请求受阻的原因 |
| `generation_audit` / `adversarial_batch` | 原有生成、复核及推理全链路记录 |

在调用前保存来源请求，调用完成后缓存实际发送上下文和响应。中断后复用已完成响应及核验结果；同一轮经验按 round_id 更新，恢复不重复增加次数。服务已经处理但响应尚未落盘的中断窗口，客户端无法确认是否完成，恢复时可能再次请求该未确认阶段。

这些统计用于改进挑战构造，不是生成器收益的因果证明。最终仍应在真实固定测试上比较效果，并报告核验接收率、有效挑战误判率、重复率及各类支持样本数。

源码：[generation/memory.py](../src/evofact/generation/memory.py)、[adversarial_batch_runner.py](../src/evofact/experiments/adversarial_batch_runner.py)。测试：[test_generator_memory.py](../tests/test_generator_memory.py)。
