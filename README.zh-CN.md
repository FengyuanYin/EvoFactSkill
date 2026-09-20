# EvoFactSkill

<p align="center">
  <b>面向虚假信息检测的自进化多智能体系统</b>
</p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/ci.yml"><img src="https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-%3E%3D3.11-blue" alt="Python >= 3.11">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</p>

EvoFactSkill 是一个用于跨领域事实核验的研究型代码库，核心是可演化的多智能体 Skill Bank。Router 构造受约束的依赖图，专家智能体输出结构化分析报告，数据集自适应 Judge 再根据当前数据集的标签空间完成判定。演化对象是完整、版本化的 Skill Package，而不是底层大语言模型的权重；候选包只有通过受保护验证后才会被提升。

> 论文元数据、预训练产物和正式结果表将在论文公开时补充。本 README 不展示尚未发布的实验结果，也不虚构论文引用。

## 核心特点

- **依赖感知的多智能体执行。** Router 返回受约束 DAG：相互独立的专家可以并行执行，存在前后依赖的专家严格按拓扑顺序执行。
- **数据集自适应判定。** 版本化标签契约把 Judge 的输出限制在 Weibo21、AMTCele、LiveFact 或 AdvFake 的原生标签空间中。`ABSTAIN` 只表示 Runtime 故障，不是数据集标签。
- **完整 Skill Package 演化。** 候选修改可以覆盖 `SKILL.md`、manifest、schema、script 和 reference，并经过内容寻址、校验、测试、门控和审计。
- **防泄漏的数据增强。** 先划分真实数据，再仅对 meta-train 执行基于真实样本的 rewriting；meta-test 和 final-test 始终保持 real-only。
- **可复现的资源核算。** 运行记录 calls、tokens、成本可用性、并发度、package digest、label-contract digest、manifest 和 trace。

## 方法概览

### 推理链路

```mermaid
flowchart LR
    A[Claim + 数据集上下文] --> B[Router / Planner]
    B --> C[受约束专家 DAG]
    C --> D1[来源可信度]
    C --> D2[时序推理]
    C --> D3[数值一致性]
    D1 --> E[结构化专家报告]
    D2 --> E
    D3 --> E
    E --> F[数据集自适应 Judge]
    G[版本化标签契约] --> F
    F --> H[预测 + 置信度 + Trace]
```

`coordinator_routing` 是 Router 使用的 Skill Package，并不存在一个独立的 Coordinator Agent。执行计划显式记录专家依赖关系：同一就绪层中的节点可以并行，后继节点必须等待其声明的前驱完成。

### 演化链路

```mermaid
flowchart LR
    A[Meta-train Traces] --> B[失败归因]
    B --> C[Package Optimizer]
    C --> D[候选 Skill Package]
    D --> E[Schema、安全与包测试]
    E --> F[固定提升门控]
    G[Real-only 受保护验证集] --> F
    F -->|通过| H[内容寻址 Package Bank]
    F -->|拒绝| I[保留当前 Champion]
```

Package Optimizer 可以为普通 Agent Package 和 generation workflow package 提出候选修改。Optimizer 自身保持固定，当前设计不启用递归的 Meta Optimizer 自演化，从而避免无限套娃和评估边界漂移。

## 安装

环境要求：Python 3.11 或更高版本。

```bash
git clone https://github.com/FengyuanYin/EvoFactSkill.git
cd EvoFactSkill
python -m venv .venv
```

激活环境并安装项目：

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
```

离线冒烟测试不需要 API Key：

```bash
evofact --config configs/dry_run.yaml dry-run --output outputs/dry-run.json
```

## 数据准备

仓库提供数据适配器，但不重新分发第三方数据集。请从数据集官方来源获取数据、遵守对应许可证，并把处理后的文件放到配置项 `data.root` 指定的位置。

| Dataset ID | Adapter | 用途 |
| --- | --- | --- |
| `weibo21` | `Weibo21Adapter` | 中文跨领域虚假信息检测 |
| `amtcele` | `AMTCeleAdapter` | 数据集标签契约评测 |
| `livefact` | `LiveFactAdapter` | 数据集标签契约评测 |
| `advfake` | `AdvFakeAdapter` | 对抗评测与演化 |

在 YAML 中选择数据集并配置领域划分：

```yaml
data:
  dataset: weibo21
  root: datasets/Weibo21
  train_domains: [domain_a, domain_b]
  final_test_domains: [held_out_domain]
  excluded_domains: [unknown]
```

正式实验前，先检查标准化样本并生成不可变的划分 manifest：

```bash
evofact --config configs/weibo21_cross_domain.yaml data inspect
evofact --config configs/weibo21_cross_domain.yaml \
  data manifest --output outputs/weibo21-manifest.json
```

Manifest 保存稳定的样本指纹，以及互不重叠的 train、evolution-validation、protected-validation 和 final-test 分区。论文结果必须连同对应 manifest 一起归档。

## 复现实验

所有全局参数必须放在子命令之前，尤其是 `--limit`、`--batch-size` 和 `--sample-concurrency`。

### 仅测试现有 Skill

在 Weibo21 上测试 100 条样本：

```bash
evofact \
  --config configs/weibo21_cross_domain.yaml \
  --limit 100 \
  --batch-size 16 \
  --sample-concurrency 8 \
  --progress \
  test --output outputs/weibo21-test.json
```

省略 `--limit` 或设置为 `0`，即可测试完整的配置分区：

```bash
evofact --config configs/weibo21_cross_domain.yaml \
  test --output outputs/weibo21-test.json
```

不修改 YAML，直接覆盖最终测试领域：

```bash
evofact --config configs/weibo21_cross_domain.yaml \
  test --final-test-domains "domain_x,domain_y" \
  --output outputs/domain-override-test.json
```

覆盖后系统会重新计算源领域和 manifest，并重新执行数据泄漏检查。

### Skill 演化

```bash
# 单轮普通演化
evofact --config configs/weibo21_cross_domain.yaml \
  --limit 400 \
  --batch-size 8 \
  --sample-concurrency 4 \
  evolve \
  --checkpoint outputs/weibo21-evolve-checkpoint.json \
  --output outputs/evolve.json

# 从最后一个原子完成的 batch 恢复
evofact --config configs/weibo21_cross_domain.yaml \
  --limit 400 \
  --batch-size 8 \
  --sample-concurrency 4 \
  evolve --resume \
  --checkpoint outputs/weibo21-evolve-checkpoint.json \
  --output outputs/evolve-resumed.json

# 只评估提案和门控，不更新 active package bank
evofact --config configs/weibo21_cross_domain.yaml \
  evolve --evaluation-only --output outputs/evolve-evaluation.json

# 跨领域元学习 episode
evofact --config configs/weibo21_cross_domain.yaml \
  meta-evolve --episodes 8 --strategy leave_one_domain_out \
  --output outputs/meta-evolve.json

# 不提出新修改，仅评估已有 checkpoint
evofact --config configs/weibo21_cross_domain.yaml \
  meta-evolve --resume --evaluation-only \
  --output outputs/meta-evaluation.json
```

这里的“训练”采用 batch 式 Skill 演化：同一个 batch 内的所有样本使用相同的 active package snapshot，只有到达批次/验证边界后才允许处理候选包，避免执行过程中发生中途漂移。它不是神经网络权重训练。

对于已经配置领域边界的数据集，`--limit` 表示**源领域训练样本的总预算**，既不是每个领域的限额，也不控制 final-test 数量。采样器使用实验 seed，将总预算尽可能平均地分配给 `data.train_domains`。Weibo21 有 8 个源领域，因此 `--limit 400` 会从每个源领域选择 50 条训练样本。如果某个领域不足其应有配额，缺额会以确定性方式分配给仍有可用样本的领域，并在输出的 `data_sampling.selected_by_domain` 中披露。

受保护的 final-test 集通过独立配置控制：

```yaml
data:
  train_sampling: balanced_by_domain
  final_test_samples_per_domain: 100
  static_test_pattern: outputs/static/{dataset}-test-{domain}.json
  require_static_test: true
```

对于 Weibo21，Runtime 会读取 `outputs/static/` 下对应历史结果中按顺序保存的 `sample_id`。每个 final-test 领域必须提供至少 100 条有效静态样本：文件恰好包含 100 条时全部保留，超过 100 条时只保留前 100 条，少于 100 条时直接报错。因此，5 个 final-test 领域会产生 500 条测试样本，而不是共同分配 100 条。如果必需的静态结果不存在、某个 ID 不属于当前加载的数据版本，或者样本不属于配置的 final-test 领域，系统同样会在推理前直接失败。这样可以保证演化系统与静态基线使用完全相同的最终测试集合。采样器本身不包含 Weibo21 专属分支；AMTCele、MCFEND 和 LiveFact 在适配器与配置提供稳定 dataset ID、domain 和对应 static-test pattern 后，可以复用同一策略。

#### 可恢复演化协议

普通演化只会在一个 batch 完整结束且已接受的 Package Bank 更新成功提交后，原子写入 checkpoint。checkpoint 保存已完成 batch 的数量与审计记录、精确的 active package 映射、累计 trace 与逐样本评估、Optimizer 和 Gate 输出，以及已经消耗的预算状态。因此，恢复后的最终评估仍覆盖中断前的样本，资源账本也不会从零开始。

恢复过程采用 fail-closed 策略。在跳过任何 batch 前，Runtime 会校验完整有效配置、配置文件摘要、数据 manifest、有序的训练/验证样本 ID、active Package Bank 和 checkpoint schema。因此，恢复命令必须与初始命令保持相同的 YAML 内容、`--limit`、`--batch-size`、`--sample-concurrency`、样本选择、数据划分和 Skill Bank。checkpoint 写入前发生中断的 batch 会重新执行，已经原子提交的 batch 不会重复执行。状态为 `complete` 的 checkpoint 是终态；`--evaluation-only` 使用隔离的临时 Bank，因而不支持恢复。

该机制提供的是 batch 边界恢复，而不是单条指令级重放。中断 batch 内已经发出但尚未形成 checkpoint 的模型调用可能在恢复后再次执行，论文报告 API 调用量和成本时应计入这一点。

#### 训练溯源与 Skill Bank 身份

每次提交的 Package Bank 都包含机器可读的训练溯源。下列标识符承担不同的科研用途：

| 标识符 | 标识范围 | 科研用途 |
| --- | --- | --- |
| `training_run_id` | 一次具体的演化运行；恢复时保持不变 | 区分使用相同实验配置重复执行的多次训练 |
| `identity_digest` | 有效配置、配置文件、manifest、训练/验证样本集合和 limit | 判断两个 run 是否属于同一实验设计 |
| `bank_digest` | 一组完整且内部一致的 Package Bank | 精确选择和引用推理所用的 Router/Specialist/Judge 系统 |
| `package_digest` | 单个内容寻址的 Skill Package | 审计某个 Agent Package 的全部文件内容 |
| 语义版本号 | 便于人工阅读的 Package 发布版本 | 描述版本变化，但不能单独用于精确复现 |

训练溯源还包含完整有效配置、配置文件摘要、manifest ID、split 指纹和样本数、有效配置中的随机种子、起始 Bank digest、起始 package 的精确 digest 与版本，以及逐 batch 的 Proposal/Gate 审计。因此，正式实验至少应同时报告源码 commit、数据 manifest ID、`training_run_id`、`identity_digest` 和最终 `bank_digest`。

使用以下命令检查 active bank 和不可变事务历史：

```bash
evofact --config configs/weibo21_cross_domain.yaml skills banks
evofact --config configs/weibo21_cross_domain.yaml skills bank-show active
evofact --config configs/weibo21_cross_domain.yaml \
  skills bank-export active --output outputs/weibo21-skill-bank-lock.json
```

`skills bank-show` 会输出 package 版本、精确 digest、可用的来源 run 和训练溯源。`skills bank-export` 会生成 `skill_bank_lock_v1` 锁文件。锁文件记录的是身份映射，不会复制 Package 内容，因此对应的内容寻址 blob 必须继续保存在当前配置的 `skill_store` 中。

#### 版本锁定推理

推理默认使用 active bank。对于论文实验或重复实验，建议导出锁文件并与实验产物一起保存。将 `--skill-bank` 放在子命令之前，可以使用训练 run ID、bank digest 或锁文件恢复一组内部一致的 Router/Specialist/Judge：

```bash
# 推荐：使用导出的不可变锁文件
evofact --config configs/weibo21_cross_domain.yaml \
  --skill-bank outputs/weibo21-skill-bank-lock.json \
  test --output outputs/weibo21-locked-bank-test.json

# 仓库仍保留历史记录时，也可以直接使用 run ID
evofact --config configs/weibo21_cross_domain.yaml \
  --skill-bank evolve-<identity>-<run> \
  test --output outputs/weibo21-run-id-test.json

# 或使用完整 Bank digest
evofact --config configs/weibo21_cross_domain.yaml \
  --skill-bank <bank-digest> \
  test --output outputs/weibo21-bank-digest-test.json
```

测试报告会嵌入最终解析得到的 bank digest、语义版本、package digest 和可用训练 provenance。项目刻意不支持单独覆盖某一个 Agent Package，因为混用不同训练快照会破坏系统级实验身份。引入溯源机制之前创建的历史 Bank 仍然可以加载，但其缺失的 provenance 无法事后重建，报告相关结果时必须明确披露这一限制。

### 统一配对消融

```bash
evofact --config configs/weibo21_cross_domain.yaml \
  ablation \
  --arms "full,no-evolution,instructions-only,no-discovery,rule-proposer" \
  --seeds 5 \
  --bootstrap-iterations 2000 \
  --output outputs/evolution-ablation.json
```

每个实验臂从相同的种子包开始，使用相同 manifest 和受保护的 final-test 样本顺序，并且只写入隔离的临时 Package Bank。报告包含实际生效的机制配置、逐 seed 指标、McNemar 检验、配对 Bootstrap 置信区间、Package Bank digest 和预算快照。`no-evolution`、`no-discovery`、`rule-proposer` 和 `no-evidence` 与 `full` 配对；`instructions-only` 与 `no-discovery` 配对，从而只改变包优化范围。`no-evidence` 仅允许用于包含证据的样本。只有 Full 配置使用 LLM proposer 时，`rule-proposer` 才是合法对照。DEMSE 另外支持 `--ablation no-negative-transfer-constraint`。

同一命令还支持 `meta-full` 搭配 `meta-no-cross-episode-aggregation`、`meta-no-negative-transfer-constraint`、`meta-no-worst-domain-constraint`、`meta-no-specialization`，以及 `generation-full` 搭配 `generation-real-only`、`generation-rule-proposer`。生成消融通过 `--facts` 提供证据事实。`unified_ablation_v2` 报告统一包含各实验族、扁平 run 列表、汇总表，以及 meta/generation 的 episode 配对 Bootstrap 区间。

### 对抗演化与 Generator 演化

```bash
# 使用标准化样本和已核验事实进行对抗演化
evofact --config configs/adversarial_llm.yaml \
  adversarial-evolve \
  --samples data/adversarial/samples.jsonl \
  --facts data/adversarial/facts.jsonl \
  --output outputs/adversarial-evolve.json

# 检查 generation audit
evofact generation audit outputs/generation-audit.json

# 由 Package Optimizer 根据 audit 提出 generation_agent 候选包
evofact --config configs/adversarial_llm.yaml \
  generator-evolve --audit outputs/generation-audit.json --propose-only

# 评估并门控序列化候选包
evofact --config configs/adversarial_llm.yaml \
  generator-evolve \
  --candidate outputs/generator-candidate.json \
  --evaluation outputs/generator-evaluation.json
```

生成样本必须保留来源 lineage，并通过 generation audit。受支持的防泄漏协议如下：

1. 首先划分真实样本。
2. 只对真实 meta-train 样本进行 rewriting。
3. 合并 real meta-train 与审核通过的 generated meta-train。
4. meta-test、protected validation 和 final-test 始终使用真实数据。

### 报告

```bash
evofact --config configs/weibo21_cross_domain.yaml \
  report --output outputs/report.json
```

为 `test`、`evolve`、`meta-evolve`、`adversarial-evolve` 指定 `--output` 后，程序会写出 UTF-8 JSON。报告包含总体指标、按领域/标签 schema 切片的指标、资源使用、trace 引用、active package digest，以及可用时的 manifest identity。

## 评估协议

主要分类指标如下：

| 指标 | 含义 |
| --- | --- |
| `accuracy_all` | 在全部有标签样本上的准确率，Runtime 故障也计入分母 |
| `macro_f1_all` | 在全部有标签样本上的 schema-aware Macro-F1 |
| `coverage` | 得到合法、非 Runtime 数据集标签的样本比例 |
| `covered_accuracy` | 仅在已覆盖样本上的准确率 |
| `selective_risk` | `1 - covered_accuracy` |
| `ece` | Expected Calibration Error |
| `brier` | 仅对声明正类的二分类 schema 计算的 Brier Score |
| `evidence_coverage` | 系统实际获得证据的样本比例 |
| `mean_cost` | 存在有效价格表时的平均后端成本 |

报告 `accuracy_all` 和 `macro_f1_all` 时必须同时报告 `coverage`。如果 Runtime 拒绝了大量困难样本，单独的高 `covered_accuracy` 没有意义。`evidence_coverage = 0` 表示无外部证据设置，不能把该结果描述为 evidence-grounded verification。

`ABSTAIN` 只表示预算耗尽、超时、必需报告缺失或 Judge 输出违反契约等 Runtime 结果。它不会作为合法标签交给 Judge。标签顺序、原生标签映射和可选正类由版本化数据集标签契约定义。

对于 Weibo21 等不提供外部 `evidence` 字段的数据集，适配器会将证据规范化为空集合。Runtime 将证据评估节点标记为可选，向 Judge 传递 `evidence_mode: unavailable`，并要求 Judge 根据声明文本和成功的非证据专家报告选择数据集合法标签。缺少 evidence 本身不会触发 `ABSTAIN`；`evidence_coverage` 仍为 0，因此该结果不能描述为证据驱动核验。

## 配置与预算

主要实验控制项均位于 YAML：

```yaml
backend: openai-compatible
model: your-model-name
base_url: https://your-provider.example/v1
api_key_env: PROVIDER_API_KEY

execution:
  batch_size: 16
  max_concurrent_samples: 8
  progress: auto

evolution:
  enabled: true
  proposer: llm
  scope: package          # package | instructions
  discovery: true

budget:
  max_calls_per_run: 10000
  max_tokens_per_run: 30000000

pricing:
  provider: your-provider
  table_path: pricing/provider-date.json
  require_cost_for_promotion: true
```

生成数据对照实验可设置 `generation.use_generated_in_training: false`。此时生成与验证仍正常执行并保留审计记录，但 detector 演化只使用真实 construction 样本。`generation.proposer: rule` 可选择确定性的标签保持表述改写基线；其输出仍必须经过与 LLM 生成样本相同的 firewall 和盲审 verifier。

密钥只应保存在 `api_key_env` 所指向的环境变量中：

```bash
# Linux / macOS
export PROVIDER_API_KEY="..."

# Windows PowerShell
$env:PROVIDER_API_KEY="..."
```

Runtime 会在边界处强制执行 calls、tokens、已计价成本和并发限制。价格快照带日期；正式论文实验前必须更新并归档当时适用的 provider 价格表。

## Skill Package 与扩展新专家

种子包位于 `skills/seeds/`。一个完整 Skill Package 可以包含：

```text
skill_name/
├── SKILL.md
├── metadata.json
├── scripts/
├── references/
└── schemas/
```

专家由通过校验的 package manifest 和 report schema 动态发现，不依赖硬编码的 Python 角色列表。新增专家时，需要声明 kind、scope、triggers、能力契约、instruction entry point 和结构化输出 schema；当契约与 Claim 和数据集上下文匹配时，Router 就可以选择该专家。

常用检查命令：

```bash
evofact skills list
evofact package validate skills/seeds/temporal_reasoning
evofact package test skills/seeds/temporal_reasoning
evofact package diff path/to/before path/to/after
```

## 仓库结构

```text
EvoFactSkill/
├── configs/              # 实验配置
├── pricing/              # 带日期的模型服务价格快照
├── skills/
│   ├── seeds/            # Router、专家、Judge、Generator、Optimizer 初始包
│   └── store*/           # 内容寻址的演化 Package Bank
├── src/evofact/
│   ├── data/             # Adapter、标签契约、Manifest、泄漏检查
│   ├── runtime/          # 路由、DAG 执行、后端、预算、Trace
│   ├── evolution/        # 失败归因与完整 Package 优化
│   ├── generation/       # Rewriting、验证与 Generation Audit
│   ├── experiments/      # Test、演化、元学习、对抗实验 Runner
│   ├── validation/       # 固定门控、校准与迁移检查
│   └── reporting/        # 机器可读实验报告
└── tests/                # 离线单元、集成与契约测试
```

## 测试

测试套件使用注入的 Fake Backend，离线运行，不会调用付费 API：

```bash
python -m pytest
python -m ruff check src tests
```

使用真实后端时，应先通过较小的 `--limit` 和严格预算验证全链路。Mock Backend 与 dry-run fixture 的输出不能被当作论文性能结果。

## 适用范围与局限

- 本项目演化 Skill Package，不训练基础模型权重。
- Package Optimizer 保持固定，目前未实现 Optimizer 自演化。
- 仓库不内置检索系统；证据快照必须由数据集或外部 workflow 提供，无证据结果必须明确披露。
- 统一消融运行器已在同一报告契约下覆盖普通 Skill 演化、DEMSE 控制项和对抗生成控制项。
- 数据集获取、许可证合规和处理版本哈希由实验者负责。
- 真实后端结果依赖模型版本、限流策略和带日期的价格快照。

## 引用

论文公开后将在此补充正式 BibTeX。在此之前，请引用仓库 URL，并记录实验所使用的精确 commit。

## 许可证

项目代码采用 [MIT License](LICENSE)。第三方数据集和模型服务仍受各自条款约束。
