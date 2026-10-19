# Optimizer 跨批记忆与上下文窗口管理

本功能仅接入 `feedback-evolve`、`meta-evolve` 和 `adversarial-evolve` 的批处理优化器。普通 `evolve` 的调用与提示保持原样。默认关闭，已有实验配置、技能库和结果保留；启用后建立新实验，不混接旧断点。

## 启用与配置

```yaml
optimizer_memory:
  enabled: true
  context_window: null
  reserved_output_tokens: 8192
  safety_fraction: 0.1
  max_overflow_retries: 2
  tokenizer_encoding: null
```

`context_window: null` 时，从实际 optimizer 服务的 `/models` 元数据读取所选模型的 `context_window`，并缓存到 checkpoint。不是读取检测器模型的窗口。服务不提供有效值时，在真实训练推理之前保存 `optimizer_context_failure` 并停止，不猜测模型窗口。

只有有效的窗口结果才作为后续查询缓存。临时超时或服务故障恢复后，可使用原配置 `--resume` 重新查询，不会永久使用失败时的空结果。

接口不支持元数据时，在新实验准备命令追加 `--optimizer-context-window` 和服务商确认的正整数窗口。该显式设置优先于自动发现；不要根据模型名字猜值。

每次请求按如下方式分配：

```text
可用输入 = floor(窗口 × (1 − safety_fraction)) − 预留输出
```

若 optimizer 后端已设置输出上限，按该上限预留；否则按 `reserved_output_tokens` 预留，并在这三个路线的受管理 optimizer 请求中设置相同输出上限。服务元数据声明更小的输出上限时相应下调。普通推理与普通 evolve 的输出设置不改变。

设置 `tokenizer_encoding` 且安装了可用的 `tiktoken` 时，按指定编码计数；必须由使用者确认编码匹配服务模型。未配置或不可用时，用 UTF-8 字节数加协议余量保守估算，审计标记为 `conservative_utf8_bytes`。它不是精确 tokenizer，可能提前压缩上下文。实际输入用量高于估算时，只向上校准后续估算系数。

## 新实验启动

离线准备带时间标签的三条路线各两折配置、32 条/域的独立候选池及每次 4 条/域的验证设置：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_independent_validation_run.py --samples-per-domain 32 --evaluation-samples-per-domain 4 --optimizer-memory
```

本次已准备的目录为：

```powershell
$runRoot = 'docs/independent-validation/20261019-110801-4fcddff6'
```

训练和测试沿用 [独立均衡验证启动教程](独立均衡验证实验启动教程-Windows.md) 的循环，将 `$runRoot` 换成上面的新目录。`$route` 选择三条路线之一。准备脚本不调用模型；启动训练后才可能查询模型元数据及发生模型调用。重新执行准备脚本会产生另一个目录，应使用新输出的路径。

## 记忆保留什么

每次完成的验证尝试生成一条结构化观察，包含批次、尝试、修改假设、目标技能、版本与摘要、接受/拒绝原因、发现验证的样本量和整体指标。无修改、非法提案、超时等未形成候选的尝试也记录状态与原因。

完整已消费训练 trace、baseline/candidate 的发现验证 trace、提案和原始请求继续落盘。它们不因一次请求放不下而被删除。实际请求默认使用程序生成的摘要，不额外调用 LLM 总结历史；相关历史可检索一条代表 trace 补充诊断。

经验按目标、失败类型与处理结果分组，保留观察次数、来源 ID 和版本。它们是历史观察，不能把一次小样本拒绝视为永远无效的结论。当前技能内容始终是修改依据，历史技能版本不会覆盖当前包。

独立验证仅提供拒绝原因，不把独立样本、标签、逐样本预测或指标明细写入可见记忆；final-test 数据继续隔离。历史摘要及检索 trace 发送前仍检查 held-out 重合，避免已消费历史中与当前受保护数据重合的内容进入新请求。

## 超出窗口时保留哪些内容

首先去掉完全重复的本批 trace，然后按相关性、重复次数及时间组织历史。相同优先级中，旧记忆先退出本次请求；旧而反复出现的相关问题可以优先于最近的无关问题。

缩减依次移出旧检索 trace、较早的完整拒绝反馈、低优先级历史摘要、旧提案，然后将最近被拒包的文件正文退化为文件路径/摘要清单，再减少最近反馈和本批 trace 的重复案例。最近反馈至少保留每侧一条 trace，本批至少保留一条；原始被拒包仍保留在审计。修改规则、输出协议及当前目标包不按字符截断。

如果完整技能库仍放不下，进入两阶段：

1. 发送可编辑技能目录、trace 与可容纳的记忆，模型选择编辑目标、新增 specialist 或不修改。
2. 编辑时加载目标包的完整文件；新增时提供对应上下文；重新按 token 预算组装记忆并产生正式提案。

每个阶段和每次重试分别计入调用与用量预算。选择目标不等于绕过补丁、摘要、安全或验证检查。若连目标完整包、协议和最小诊断都放不下，保存并停止，不截掉 SKILL.md 内容或 digest 来凑窗口。

选择了未知目标、非法动作或提案偏离所选目标，属于 `OptimizerProtocolFailure`：记录为一次失败尝试，继续使用本批剩余尝试次数；不会误报为窗口不足而停训。达到每批上限后仍按原规则进入下一批。

## 重试与断点恢复

只有明确的 HTTP 上下文长度错误触发缩减后的有限重试。网络超时、429、认证错误、JSON 解析错误和输出截断分别记录；不会因这些错误自动删除旧记忆并不断重试。

输出截断表示服务在完成 optimizer JSON 前停止，它与输入超限是不同事件。需要结合输出预留和候选修改规模判断，当前不把它自动归因于旧记忆过多。

服务已返回 usage、随后输出截断或 JSON 解析失败时，实际 token 和可定价费用仍计入账本。没有 usage 的失败只记录已发出的调用与用量未知状态，不编造费用。修复前没有保存的历史 usage 无法自动补回。计数校准会在已有系数上继续按实际低估比例向上修正。

checkpoint 保存当前请求的源摘要、模型窗口、有效 payload、已移出条目、预算、系统提示摘要和响应。完成的目标选择与提案响应可直接复用；续跑不重新选记忆或目标。若进程恰好在服务已处理、响应尚未落盘之间中断，客户端无法确认服务处理结果，恢复可能再次请求该未确认阶段。

## 审计字段

| checkpoint 字段 | 内容 |
| --- | --- |
| `optimizer_memories` | 跨批观察及已消费的原始诊断记录 |
| `optimizer_requests` | 已处理请求的完整审计及最终解析结果 |
| `optimizer_request` | 尚未完成或受阻的请求，含各阶段缓存 |
| `optimizer_capabilities` | 实际 endpoint/model 对应的模型窗口元数据 |
| `optimizer_token_factor` | 使用实际用量向上校准的计数系数 |

每个阶段的 `calls` 包含估算输入、可用输入、输出预留、计数方式、payload/system 摘要、调用状态与实际 usage；异常还包含失败分类。`omitted` 记录哪些可选内容退出该次请求。

状态为 `optimizer_context_failure` 时，CLI 返回非零退出码并保留断点；先查看错误与请求阶段。修改窗口或模型配置属于新实验身份，不应强行接到旧配置断点上。

源码入口：[optimizer_memory.py](../src/evofact/evolution/optimizer_memory.py)、[MetaBatchRunner](../src/evofact/experiments/meta_batch_runner.py)。测试入口：[test_optimizer_memory.py](../tests/test_optimizer_memory.py)。
