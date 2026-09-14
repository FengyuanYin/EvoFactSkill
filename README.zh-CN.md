# EvoFactSkill

[English](README.md) | **简体中文**

[![CI](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/ci.yml)
[![Security](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/security.yml)
[![PyPI](https://img.shields.io/pypi/v/evofactskill.svg)](https://pypi.org/project/evofactskill/)

EvoFactSkill 是一个面向**跨域与时间漂移假新闻检测**的研究框架，核心是**可验证、可自进化的技能库（Skill Bank）**。

底层语言模型始终**冻结**。真正进化的是一个可审计的外部技能库，以及围绕它的路由策略、优化器提示词与生成工作流。任何技能变更都必须先在留出域上通过实证门控才能晋级，因此技能只有在“证明自己”之后才会进入 `active`。

- Python **3.11+**，**无运行时依赖**（仅标准库）。
- 默认完全**离线**运行：确定性 Mock 后端，无需 API Key、无需联网、不接触受限数据。
- 显式配置后，同一套代码可直接驱动 OpenAI 兼容的真实模型。
- 许可证：MIT。

---

## 已实现内容

**推理与路由**

- Coordinator → Router → Specialists → Judge 推理链路，产出不可变执行 trace。
- 基于效用的规则路由，以及 `all-experts`、`random`、`static` 三种基线。
- **LLM 技能路由器**（`routing_strategy: llm`）：模型只能在合法候选技能 ID 中选择；任何协议违规——未知 ID、空选择、超出预算的选择、置信度非数值或越界——都会被拒绝，并交由规则路由器接管，同时记录回退原因。

**技能库**

- **十个种子技能**：`claim_decomposition`、`coordinator_routing`、`cross_source_contradiction`、`evidence_assessment`、`judge_decision`、`linguistic_manipulation`、`numerical_consistency`、`skill_optimizer`、`source_credibility`、`temporal_reasoning`。
- 内容寻址的版本化仓库，技能状态包括 `candidate`、`active`、`pareto`、`frozen`、`retired`。
- 八种生命周期操作：`ADD`、`EDIT`、`SPLIT`、`MERGE`、`GENERALIZE`、`SPECIALIZE`、`RETIRE`、`ROLLBACK`。

**进化机制**

- 规则式**错误归因**，覆盖八类错误（`routing_miss`、`evidence_miss`、`evidence_hallucination`、`temporal_leakage`、`reasoning_error`、`judge_aggregation_error`、`label_mapping_error`、`abstention_error`），配合稳定的错误聚类、反事实贡献接口与标签脱敏的经验蒸馏。
- **LLM 优化器**（`evolution.proposer: llm`）：把归因报告转化为受约束的 `add` / `edit` / `no_change` 决策，并保留规则 proposer 作为回退。
- **DEMSE**（Domain-Episodic Meta-Gated Skill Evolution，域情景元门控技能进化）跨域元演化。
- **对抗生成链路**：成功/失败 trace → LLM 自主选择策略 → 生成样本 → 独立审核 → DEMSE 检测技能门控。

**评测、数据与安全**

- 成对重复评估：bootstrap 置信区间、McNemar 检验、覆盖率与成本约束、受保护域回归检查以及 Pareto 保留。
- 指标把 `ABSTAIN` 计入主分母，并单独报告覆盖率、选择性风险、ECE、Brier，以及按域/按时间窗口的细分结果。
- 数据适配器：**Weibo21**、**AMTCele**、**LiveFact**、**AdvFake**，配套可复现清单与泄漏检测。
- 八条消融臂：`single-llm`、`static`、`all-experts`、`random`、`prompt-only-evolution`、`no-discovery`、`no-negative-transfer`、`full`。
- 基于 AST 的可执行技能筛查。脚本类技能一律需要人工复核，除非额外提供更严格的外部沙箱。

## 快速开始——离线、无需 API Key

安装到本地环境（需要本地 setuptools；`--no-build-isolation` 可保持离线）：

```powershell
python -m pip install -e . --no-build-isolation
```

或者不安装，直接从源码树运行：

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m evofact.cli --config configs/dry_run.yaml dry-run
```

安装后控制台入口为 `evofact`（映射到 `evofact.cli:main`）。

与 CI 完全一致地运行离线测试：

```powershell
python -m pytest
```

`configs/dry_run.yaml` 中特意保留了一个困难样本，以便观察错误归因与候选校验的行为。

## 命令行

全局参数（`--config`、`--dataset`、`--data-root`、`--limit`）必须写在子命令**之前**。所有命令均以 JSON 输出结果。

```text
evofact data inspect
evofact data manifest [--output manifest.json]

evofact dry-run                 # 在训练夹具上离线推理
evofact test                    # 在最终测试划分上离线推理
evofact evolve                  # 固定验证门控的闭环进化
evofact validate                # 仅输出门控决策
evofact ablation                # 运行八条消融臂
evofact report                  # 多种子报告（JSON + Markdown + CSV）

evofact meta-evolve             # DEMSE 跨域元演化
  --final-test-domains outer_holdout
  [--episodes N] [--strategy repeated_holdout|leave_one_domain_out]
  [--meta-test-domain-count N] [--resume] [--evaluation-only]
  [--ablation no-cross-episode-aggregation|no-worst-domain-constraint|no-specialization]

evofact adversarial-evolve      # trace → 生成 → 审核 → 门控
  [--samples samples.jsonl] [--facts facts.jsonl]
  [--final-test-domains outer_holdout] [--resume] [--evaluation-only]

evofact skills list
evofact skills show NAME [--snapshot HASH]
evofact skills diff NAME [--snapshot HASH]
evofact skills freeze NAME
evofact skills retire NAME
evofact skills rollback NAME --snapshot HASH
```

从源码运行时，给每条命令加前缀：

```powershell
$env:PYTHONPATH='src'
python -m evofact.cli --config configs/default.yaml <command>
```

`meta-evolve` 与 `adversarial-evolve` 支持 `--evaluation-only`：只构建推理运行时，绝不提交技能库，适合机制验证。`--resume` 仅在配置、数据与 SkillBank 指纹全部一致时才会恢复。

## 配置

配置为纯 YAML，由标准库实现的轻量加载器（`evofact.config`）解析。

| 配置文件 | 后端 | 用途 |
|---|---|---|
| `configs/dry_run.yaml` | mock | 最小的离线冒烟运行，测试与文档均使用它。 |
| `configs/default.yaml` | mock | 默认离线配置。 |
| `configs/demse_dry_run.yaml` | mock | DEMSE 元演化夹具（五个源领域 + 一个外层留出领域）。 |
| `configs/llm_router.yaml` | openai-compatible | LLM 路由器 + LLM 优化器端到端，需要 `DEEPSEEK_API_KEY`。 |
| `configs/adversarial_llm.yaml` | openai-compatible | 对抗生成链路，需要 `DEEPSEEK_API_KEY`。 |

关键配置项：

```yaml
backend: mock                 # mock | openai-compatible
model: mock-v1
api_key_env: DEEPSEEK_API_KEY # 唯一出现凭据名称的位置

routing_strategy: utility-aware   # utility-aware | all-experts | random | static | llm
max_skills_per_item: 3

evolution:                    # 技能提案的生成方式
  proposer: rule              # rule | llm
  fallback_to_rule: true      # LLM 路径失败时回退到规则 proposer
  optimizer_skill: skill_optimizer
  max_reports: 12
  max_skills: 12
  max_text_chars: 8000
  max_instructions_chars: 60000
  max_total_chars: 300000

gate:                         # 晋级要求
  repeats: 3
  min_macro_f1_gain: 0.01
  min_coverage: 0.8
  max_protected_domain_drop: 0.02
  alpha: 0.05
  max_cost_ratio: 1.5
```

`meta_learning.*` 用于配置 DEMSE 的 episode 计划与额外约束；`generation.*` 用于配置对抗生成链路（批大小、trace/文本上限、探测比例、审计路径）。

## 数据集

数据集**不会被复制或再分发**。请在本地配置的 `dataset_roots` 中指向已授权、只读的位置：

```yaml
dataset_roots:
  weibo21: F:/datasets/Weibo21
  amtcele: F:/datasets/AMTCele
  livefact: F:/datasets/LiveFact
  advfake: F:/datasets/AdvFake
```

`evofact data inspect` 会独立报告每个适配器；缺少某个数据集不会阻塞离线测试，也不影响其他适配器。

划分策略：

- 官方测试记录与 AdvFake 仅作测试用途。
- LiveFact `+3` 为测试，`0` 为受保护验证集，更早的快照作为训练/进化素材。
- 事件 ID 以原子方式分配到唯一划分。
- 归一化内容重复、事件重叠、以及晚于声明截止时间的证据都会作为泄漏上报。

正式论文实验前，请确认数据集许可、标签语义与所使用的具体处理版本。生成的清单会保存稳定的样本指纹，必须与实验结果一并归档。

## 系统结构

```text
                      ┌──────────────── routing_strategy ────────────────┐
sample ──▶ Coordinator ──▶ Router ──▶ Specialists（并行）──▶ Judge ──▶ 决策（含 ABSTAIN）
                      └── utility-aware | all-experts | random | static | llm
                                        │
                                        ▼
                                  不可变执行 trace
                                        │
   ┌────────────────────────────────────┴─────────────────────────────────┐
   ▼                                                                      ▼
固定门控路径                                                        DEMSE 元演化路径
错误归因（8 类错误）                                                域情景 episode
   → 聚类                                                           meta-train → 候选技能
   → 提案：规则 proposer | LLM 优化器                                meta-test → 迁移效用
   → 技能包与脚本安全扫描                                            按语义身份聚合
   → 成对重复验证                                                    门控：泛化/特化/pareto/拒绝
   → active | pareto | review_required | rejected
```

最终测试命令只构建推理运行时，绝不实例化进化引擎。测试标签永远不能用于生成或选择技能。

### LLM 优化器

将 `evolution.proposer` 设为 `llm` 即启用 `SkillOptimizerAgent`（`src/evofact/evolution/optimizer.py`）：

1. 归因报告、trace 与当前技能库被蒸馏为**不含真实标签、且长度受限**的上下文（报告数、文本与指令长度上限均可配置）。
2. 提示词就是 `skill_optimizer` 这个种子技能本身——按名称从技能库中解析——因此优化器自身的指令是可进化、可评审的产物，而不是硬编码文本。
3. 模型必须返回单条受约束决策：`add`（仅允许新增 specialist）、`edit`（可针对 specialist/router/judge，且不允许重命名）、或 `no_change`。必须给出理由与置信度；字段非法或未知一律拒绝。
4. `no_change` 不产生任何提案；其余决策仍须通过与规则路径完全相同的安全扫描与验证门控。
5. 当 `fallback_to_rule: true` 时，任何失败都会回退到规则 proposer，而不是静默丢弃该聚类。

## 指标与解读

`accuracy_all` 与 `macro_f1_all` 使用全部有标签样本，因此 `ABSTAIN` 不会被悄悄剔除。`covered_accuracy` 单独报告，必须始终与 `coverage`、`selective_risk` 一起解读。

晋级不能只靠点估计为正：候选技能必须同时满足最小增益与覆盖率、受保护域回归、成本、安全以及成对统计支撑等条件。有实际价值但尚不可晋级的非支配候选，可以保留在 Pareto 归档中。

## 真实模型后端

标准库实现的 OpenAI 兼容后端可直接用于集成。API Key **只能**通过配置的环境变量提供——绝不写入 YAML、源码、trace 或报告。

```powershell
$env:DEEPSEEK_API_KEY='...'
$env:PYTHONPATH='src'
python -m evofact.cli --config configs/llm_router.yaml dry-run
```

LLM 路由器提供独立自检入口，同时覆盖正常路径与非法输出的回退路径：

```powershell
python -m evofact.routing.router                 # 离线，Mock 后端
python -m evofact.routing.router --live          # 真实 API，使用 configs/adversarial_llm.yaml
python -m evofact.routing.router --config configs/llm_router.yaml --live
```

真实模型编排请先在小型 evolution-validation 清单上验证。离线测试套件绝不发起网络请求，而是注入假后端。

## 仓库结构

```text
src/evofact/
  cli.py            命令行入口
  config.py         配置数据类与 YAML 加载器
  core/             共享模型、脱敏
  data/             适配器、领域、episode、泄漏检测、清单
  routing/          规则路由、LLM 路由、路由策略
  runtime/          后端（mock、openai-compatible）、推理、trace 存储
  attribution/      错误规则、聚类、反事实贡献
  evolution/        distiller、firewall、identity、meta、operations、proposer、optimizer
  validation/       evaluator、gate、meta gate、objectives、pareto、statistics、transfer
  evaluation/       指标、校准、消融
  experiments/      runner、meta runner、adversarial runner、checkpoint、protocols
  generation/       对抗数据、生成器、提示词、划分、审核器
  reporting/        report、meta report、adversarial report
  security/         AST 可执行技能扫描器
  skills/           loader、repository、lifecycle、candidates、utility
configs/            离线、DEMSE、LLM 路由与对抗配置
skills/seeds/       十个种子技能
tests/              离线测试套件
```

## 测试与 CI/CD

CI 运行在 GitHub 托管 runner 上，所有 Action 固定到不可变 commit SHA，Dependabot 每周检查 Actions 与 CI Python 工具。

| 工作流 | 触发条件 | 要求行为 |
|---|---|---|
| `CI` | PR、推送到 `main`、手动触发、可复用调用 | Ruff lint + 格式检查、`compileall`、Python 3.11–3.14 测试、包元数据校验与干净 wheel 的 CLI 冒烟测试 |
| `Security` | PR、推送到 `main`、每周一、手动触发 | CodeQL Python 分析与 `pip-audit` 项目依赖扫描 |
| `Release` | 推送 `vX.Y.Z` tag | 复用 CI，校验 tag 与包版本一致，生成构件证明，创建 GitHub Release，再通过 OIDC 发布到 PyPI |

CI 与 Security 永远不接收模型 API Key，也不会运行真实 LLM 实验或受限数据集。成功运行会保留经校验的 `python-package` 构件七天；失败的测试矩阵只保留 JUnit 诊断文件。

在本地复现整条流水线：

```powershell
python -m pip install -r .github/requirements/ci.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
python -m twine check dist/*
```

出现失败时，请在 **Actions** 标签页打开对应 commit。测试 XML 与依赖审计 JSON 会附加到失败的运行上（若有）；CodeQL 结果见 **Security → Code scanning**；包构件附加在 `Build and verify package` 任务下。

### 一次性仓库配置

1. 在 **Settings → Environments** 中创建名称严格为 `pypi` 的环境。若发布需要人工审批，请添加 required reviewers，并把部署分支/tag 限制为匹配 `v*` 的受保护 tag。
2. 在 PyPI 创建或选择 `evofactskill` 项目。在 **Publishing** 中添加 GitHub Trusted Publisher：owner `FengyuanYin`、repository `EvoFactSkill`、workflow `release.yml`、environment `pypi`。若项目尚不存在，请使用 PyPI 的 pending publisher 流程。
3. 在 **Settings → Rules → Rulesets**（或分支保护）中保护 `main`，要求通过 PR，并将以下检查设为必需：`Lint and compile`、四个 `Test (Python 3.x)` 任务、`Build and verify package`、`CodeQL`、`Dependency audit`。
4. 启用 GitHub Actions 与 Code Scanning。若 GitHub 默认的 CodeQL 配置已启用，请先关闭，再启用本仓库的高级 `security.yml` 工作流，以免配置重复。

GitHub Secrets 中不应存放 `PYPI_API_TOKEN`、PyPI 密码、云密钥或模型凭据。只有在 `pypi` 环境放行后，发布任务才会获得一个短期 OIDC 身份。

### 发布一个版本

发布是刻意为之的动作：合并到 `main` 永远不会发布包，工作流也从不修改项目版本或自行创建 tag。

1. 在 `pyproject.toml` 与离线兼容的 `setup.py` 中把版本改成同一个 SemVer 值。
2. 运行上面的本地检查，提交版本变更并推送，等待 `main` 上 CI 与 Security 通过。
3. 创建并推送匹配的 tag（含 `v` 前缀）：

```powershell
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

发布流程会在申请写权限或 OIDC 权限之前拒绝格式错误或不匹配的 tag。wheel 与源码分发包只在被复用的 CI 工作流中构建一次；完全相同的文件会被校验和、证明、附加到 GitHub Release 并上传到 PyPI。已存在的 GitHub Release 与 PyPI 版本永不被覆盖或静默跳过。

请勿删除并重建已发布的 PyPI 版本。GitHub Environment 与 PyPI Trusted Publisher 这两项一次性设置无法安全地由仓库代码创建；在两者都配置完成并在各自设置页可见之前，CI 可用，但最终的 PyPI 发布尚未就绪。

## 研究基线

共享实验协议定义了：单 LLM、静态多智能体、全部专家、随机路由、仅提示词进化、无技能发现、无负迁移控制，以及完整系统。所有臂必须使用同一份清单、同一样本顺序与同一套指标实现。

## 现状与局限

实现审计（最近复核 2026-09-08，并针对 LLM 路由器、LLM 优化器与对抗链路做了扩展）：

- 空样本列表与空技能库保持其原有语义；只有 `None` 才会选择默认值。
- 技能的领域/数据集/时间范围是硬路由过滤条件，回退路径同样适用。
- 固定门控与 DEMSE 评测共用 split、merge、retire 的生命周期迁移；冻结目标与非法迁移一律拒绝。
- checkpoint 身份覆盖完整数据与技能内容，以及模型、随机种子、预算和门控配置。修改最终域策略同样会导致 resume 被拒。
- DEMSE 通过一次原子 active 指针更新提交完整技能库与审计事务；已提交的运行可以恢复而不重复提交。相互冲突的已接受提案需要联合评估。
- 成对检验会拒绝缺失/重复 ID 与被改动的金标签；重复的 episode 证据会被拒绝。特化不能绕过覆盖率、成本、校准或最小 episode 约束。
- `skills diff` 返回真实的快照差异；回滚必须提供快照。快照查询会拒绝类路径输入。
- 数据集报告在评估给定的最终测试划分时不会触发进化。普通推理读取已有的 active 技能库；若不存在则加载种子技能。
- LLM 路由器只在合法候选 ID 中选择，且每次被绕过都会记录回退原因；LLM 优化器的决策与规则提案一样，必须经过同一套安全扫描与门控。

已知缺口——过往的 checklist **并不**代表所有研究要求已全部完成：

- 自动提案发现目前只产出 `ADD`/`EDIT`；较慢的 Meta-Skill 循环仍是事件记录脚手架。
- 部分具名消融臂共用同一套路由实现。
- 真实后端的 token 计价，以及完整的 token/调用/并发预算约束尚未完成。
- 基于文件的技能仓库假设只有一个写入方。
- 默认结果是离线机制验证，**不是**跨域模型准确率的实证结论。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
