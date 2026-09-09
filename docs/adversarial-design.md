# Trace 驱动的单次 LLM 生成链路

当前接口仅支持 `generation.proposer: llm`。LLM 根据 meta-train 成功与失败轨迹，自主选择系统提示词中的策略、分配批次预算，在同一响应里返回完整 `samples` 和独立 `decisions`。旧的数值模板、style/delta 提议、策略权重、G0/G1 比较已删除。

## 1. 两种反馈的准确含义

| 主检测系统结果 | 对生成 agent 的含义 | 生成动作 |
|---|---|---|
| success：预测等于训练 gold | 这种类型当前难度无法让主系统犯错 | 分析有效线索，构造同类型更难、仍可由证据判定的案例 |
| failure：预测不等于 gold，包括弃权 | 这种类型仍是训练薄弱点 | 增加同类 samples，扩展措辞、实体与情境变体，加强训练 |

失败类型不要求一律升级难度；成功类型不只是重复简单正例。LLM 自主决定两类数量，程序只施加批次上限，不用预设权重选策略。`reason` 需说明生成动作和来源 trace 的关系。此处“训练”仍是以样本促成检测 Skill 的 ADD/EDIT 与验证，不执行梯度训练。

## 2. 运行链路

```text
source domains → episode meta-train / meta-test，final-test 单独隔离
meta-train → 按事件/证据/来源连通组拆 construction / probe
construction → 主检测系统推理 → 成功/失败 trace + 错误归因
system prompt（含策略）+ trace 示例 → 一次 LLM 调用 → samples + decisions
结构/来源/重复/泄漏检查 → 独立证据审核 → 合格 samples
construction 原样本 + 合格 samples → evolve_once → 检测 Skill 候选
probe → 候选效果诊断（不回传给生成器）
原始 meta-test → 配对评估 → 跨 episode 聚合 → DEMSE 门控 → 检测库
```

每个 episode 的检测基线在候选评估期间保持冻结。候选只有经过原 DEMSE 外层门控才可能生效，不能因为“生成样本很难”就晋升。生成侧既不接收 probe 文本，也不接收 probe 聚合指标。

## 3. 代码与协议

- [prompts.py](../src/evofact/generation/prompts.py)：策略列表和两份系统提示词。
- [generator.py](../src/evofact/generation/generator.py)：`build_request`、`generate`，一次请求直接返回正文。
- [verifier.py](../src/evofact/generation/verifier.py)：结构审核与独立证据判别。
- [adversarial_runner.py](../src/evofact/experiments/adversarial_runner.py)：组装、响应缓存、内层接入、诊断和审计。
- [test_adversarial.py](../tests/test_adversarial.py)：一次调用、两类反馈、输入隔离、恢复和导出验证。

请求包含 `batch_size`、`id_prefix`、完整 construction 的 `trace_counts`、最多 `max_trace_examples` 个 `examples`。每例有完整标准化 `source_sample`、脱敏的主系统 trace、训练 gold、outcome、归因。成功/失败实例交替选入上下文，某类不存在时不伪造。主检测系统自己的输入仍通过 `Sample.public_view()` 隐去标签。

响应形状（以下只展示一项以说明字段，例中文字不是实际生成结果）：

```json
{
  "samples": [{
    "sample_id": "gen-episode-1-0",
    "dataset": "weibo21",
    "text": "某项目的登记记录经过重新表述后仍报告为二十四项。",
    "label": "REAL",
    "domain": "science",
    "event_id": "event-1",
    "published_at": null,
    "evidence": [{"text": "该项目登记数量为24项。", "source": "archive://record-1", "published_at": null, "stance": "unknown"}],
    "metadata": {}
  }],
  "decisions": [{
    "sample_id": "gen-episode-1-0",
    "source_sample_id": "source-1",
    "strategy": "success_boundary_extension",
    "reason": "成功类型升难度：改变数量表达，减少原文字面匹配线索。",
    "source_trace_ids": ["trace-1"]
  }]
}
```

`dataset/domain/event_id/published_at/evidence` 必须与一个已提供的 source_sample 一致；`text` 由 LLM 自由撰写，不再套固定模板。实际来源文本的语言、体裁、长度示例通过 source_sample 传入。这里的样式指项目标准化 Sample schema，不直接输出各原始数据集的 CSV/Excel 私有列；如需原生文件，需另外实现导出适配器。

## 4. 审核与数据边界

结构审核检查完整字段、batch 上限、唯一 ID、非空正文、正文长度、REAL/FAKE 标签、已知来源、已知策略、来源 trace 白名单及空 metadata。复制训练原文、批内重复或带留出内容的条目拒绝。证据只能来自 construction，不允许模型创造证据来源。

独立审核请求只包含 sample_id、text、evidence，看不到生成标签、策略、gold 和成功/失败。它返回 REAL/FAKE/UNKNOWN 与理由；只有与生成标签一致的条目进入训练。源证据 stance 是针对原断言的，因此进入新样本时清为 unknown。没有证据或证据不足时拒绝，不能把生成者的自评当 gold。

“一次”限定生成请求。结构合格条目另需一次独立 LLM 审核；没有结构合格条目则跳过。检测、归因重放和 probe/meta-test 评估也有自己的调用。没有自动补生成或修复请求；整个响应 schema 错误会明确失败，单条校验失败记录拒绝，其余条目继续。

事件、证据文本、来源跨域共享会在推理前报错。source 请使用具体快照或文档标识，使用整站同一个 URL 会使分组过粗。每域至少两个独立组，probe 需含两类标签。meta-test/final-test 的文本、标签和轨迹均不得进入生成上下文。

## 5. 配置与运行

使用 [adversarial_llm.yaml](../configs/adversarial_llm.yaml)，在本地环境配置 `DEEPSEEK_API_KEY` 后运行：

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m evofact.cli --config configs/adversarial_llm.yaml adversarial-evolve --samples samples.jsonl --final-test-domains outer_holdout --evaluation-only
```

`--samples` 使用 [load_samples](../src/evofact/generation/data.py) 支持的标准化 JSONL，并保留 evidence 快照。`--facts` 可选，用于额外检查数值事实链接，已不决定生成正文。也可使用已有 `--dataset/--data-root` 适配器，但缺少 evidence 的数据无法通过生成标签审核。不提供输入时使用模拟夹具；**生成和审核仍调用真实 LLM**。

`batch_size` 是每个 episode 的 samples 总数上限（不是对数）；`max_trace_examples` 是上下文实例上限；`max_text_chars` 限制单条生成正文长度；`probe_fraction` 是组级划分比例。旧 `rounds/pairs_per_round/mode/learning_rate` 配置不再支持。

离线机制验收：

```powershell
python -B -m unittest discover -s tests -p test_adversarial.py -v
```

测试注入 FakeJSONBackend，生产命令没有 deterministic/mock 生成回退。普通 `dry-run` 和 `meta-evolve` 的离线入口保留。

## 6. 适应、持久化和恢复

生成 agent 的适应是“当前 trace → 本批策略和数量”，系统 prompt 固定、模型参数固定；没有可学习权重库，也不自动改写系统提示词。每次新的有效运行根据当前检测库重新取 trace；当检测 Skill 更新，新的成功/失败分布会影响下次出题。一个 DEMSE run 的各 episode 不累积跨域生成记忆，以保护留出域。

生成响应返回后立即保存到 checkpoint 同级 `generation-cache`。缓存键覆盖数据/episode、配置、两份系统提示词、construction、检测库；`--resume` 可复用它继续审核。已完成 episode 直接恢复完整审计，不重做生成。若 API 返回前网络失败，或返回后尚未成功落盘就崩溃，不能保证服务端全局 exactly-once；本地已保存的响应不会主动再生成。

生成正式存储为 `generation_audit_v2`，按 run_id 保存 prompt_hash、请求、原始响应、审核、合格样本、拒绝理由和 probe 诊断；它是审计档案，不是策略训练库。检测库和生成审计各自原子写入；若后者失败，使用相同参数 `--resume` 补齐。evaluation-only 仅写缓存、checkpoint 和报告，不提交正式两处状态。

旧模板版本的 checkpoint/生成 bank 不兼容。新配置使用 `outputs/adversarial-llm`；不要复用旧路径强行恢复。

## 7. 输出与研究解释

[报告器](../src/evofact/reporting/adversarial_report.py) 输出原 DEMSE 报告、adversarial_report.json/md，以及每个 episode 的 samples JSONL。按 episode 单独导出，避免把别的 episode 的训练样本放进当前留出域；审计字段不进入样本 metadata。

difficulty 是合格生成样本上主系统错误/弃权比例；probe 的 teaching_gain 是单个候选相对基线的 F1 差，只作诊断，不参与生成策略更新。generator_calls 表示该运行已保存的逻辑生成调用数，恢复时显示历史计数，不能解释为本次进程新发出的请求数。

当前实现没有真实模型有效性实验。独立 LLM 审核仍可能出错；更高难度、同类增样和检测泛化提升需要真实数据验证，不能以离线替身结果或生成数量证明效果。
