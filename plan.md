# EvoFactSkill Plan

## 挑战链路架构：一次 LLM 响应（2026-09-09）

`generation/prompts.py` 保存可选策略及生成/独立审核系统提示词；`generator.py` 汇总 construction trace，交替选取成功/失败实例，单次调用后端返回 `samples + decisions`。成功升难度、失败同类增样，预算分配由模型完成。

`verifier.py` 先检查完整 Sample schema、来源字段、重复、预算、留出数据和 decisions 的 trace 白名单，再独立调用 LLM 仅根据正文和来源证据判断 REAL/FAKE/UNKNOWN。只有与生成标签一致的样本进入检测训练；不通过时拒绝，不渲染模板、不重试生成。证据 stance 清为 unknown，避免把源断言结论迁移到新断言。

`AdversarialEvolutionRunner._evolve_episode`：按事件/证据来源拆 construction/probe → 冻结检测基线 → construction 推理与归因 → 一次生成及缓存 → 审核 → construction+合格 samples 调用 evolve_once → probe 诊断 → 原 DEMSE 外层。每个 episode 独立，不传播其他 episode 的生成记忆。

检查点目录下的 generation-cache 保存完整已返回响应，key 覆盖配置、prompt、数据/episode、construction 和检测库。完成记录由原 checkpoint 保存；正式 generation.store_path 是版本化运行审计，不再是可学习权重库。报告按 episode 导出标准化 samples JSONL，防止混合留出域。检测提交与生成审计各自原子；相同参数 --resume 可补齐中断提交。

## 2026-09-08 实现补强

候选评估与提交共用纯函数 `apply_candidate`，按操作移除全部目标再加入候选；仓库以一次 active 指针替换提交完整结果。恢复身份覆盖完整数据与技能内容、模型、seed、预算、门控及领域计划；已提交事务用 run ID 去重。路由将作用域作为硬过滤，特化不得绕过覆盖、成本和校准限制。新增回归测试作为本次验收依据；旧勾选记录仅代表历史验证。

> DEMSE 修订：本计划新增 Domain-Episodic Meta-Gated Skill Evolution 实现。原固定验证闭环保留为兼容模式与实验基线；DEMSE 不更新底层模型参数，而在离散 SkillBank 空间执行域情景双层元优化。

## 架构概览

系统采用六层架构：

```text
Dataset & Split Layer
        ↓
Inference Runtime
Coordinator → Skill Router → Specialists → Aggregator → Judge
        ↓
Trace & Attribution Layer
错误分类 + 反事实 Skill 贡献估计
        ↓
Evolution Layer
经验蒸馏 + ADD/EDIT/SPLIT/MERGE/GENERALIZE/SPECIALIZE/RETIRE
        ↓
Validation & Governance Layer
配对重复评估 + 多目标门控 + 负迁移检查 + 人工审核
        ↓
Versioned SkillBank
Active / Candidate / Pareto Archive / Retired / Rollback
```

主要组件：

- `DataRegistry`：统一四类数据集格式，生成训练、进化验证、保护域验证和最终测试清单。
- `InferenceRuntime`：通过统一模型接口运行单 LLM 或多智能体实验臂，确保所有实验共用同一输入和指标口径。
- `SkillRouter`：结合内容特征、适用范围、历史效用和预算选择 Skill。
- `TraceStore`：保存逐样本不可变 trace，并禁止将测试标签写入可供进化器读取的区域。
- `AttributionEngine`：执行规则归因和可选反事实重放，定位路由、专家、证据或 Judge 错误。
- `ExperienceDistiller`：聚合成功模式和错误簇，生成紧凑经验摘要。
- `EvolutionEngine`：产生结构化生命周期操作，不直接修改 active Skill。
- `ValidationGate`：在固定分层样本上比较 baseline/candidate，执行统计、回归和安全检查。
- `SkillRepository`：以内容寻址快照保存版本、父子关系、diff、效用和 Pareto 状态。
- `UtilityTracker`：更新 Skill 的使用率、边际收益、成本和负迁移记录。
- `ExperimentRunner`：组织训练、进化、验证、测试、消融、断点续跑和报告。
- `MetaEvolutionLoop`：默认关闭，以较慢周期进化 Attribution、Distillation 和 Proposal 策略。

依赖方向保持单向：数据与领域模型位于底层，推理不依赖进化模块，最终测试入口不会初始化 EvolutionEngine。

### DEMSE 域情景元进化控制层

```text
外层最终测试域（全程隔离）
             │
源领域池 ──→ DomainEpisodeSampler
             │
     ┌───────┴────────┐
     │                │
meta-train 域     meta-test 域
     │                │
归因、蒸馏、候选生成   baseline/candidate 只读配对评估
     │                │
     └──── EpisodeEvaluation
                 │
      CrossEpisodeAggregator
                 │
          MetaValidationGate
                 │
 GENERALIZE / SPECIALIZE / PARETO / REJECT / RETIRE
                 │
       SkillRepository 原子提交
```

- `DomainEpisodeSampler` 负责域级互斥划分、轮换覆盖与确定性 episode ID。
- `CandidateFirewall` 保证候选生成器仅接收 meta-train 脱敏信息。
- `MetaEpisodeExecutor` 冻结 baseline/candidate 快照并在 meta-test 域执行严格配对评估。
- `CrossEpisodeAggregator` 按候选语义指纹归并并累计迁移证据。
- `MetaValidationGate` 输出泛化、特化、Pareto、拒绝、淘汰或人工审核决策。
- `MetaEvolutionRunner` 仅在全部 episode 聚合结束后集中提交 SkillRepository。

## 核心数据结构

核心对象采用不可变数据记录；持久化统一使用带 Schema 版本的 JSON/JSONL。

```python
@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset: str
    text: str
    label: str | int | None
    domain: str | None
    event_id: str | None
    published_at: datetime | None
    evidence: tuple[Evidence, ...]
    metadata: Mapping[str, Any]
```

`Sample.public_view()` 返回删除 label 和泄漏字段后的推理输入。

```python
@dataclass(frozen=True)
class DataManifest:
    manifest_id: str
    dataset_fingerprints: Mapping[str, str]
    train_ids: tuple[str, ...]
    evolution_validation_ids: tuple[str, ...]
    protected_validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    split_policy: Mapping[str, Any]
```

Manifest 一经用于实验不得原地修改。

```python
@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    name: str
    kind: Literal["router", "specialist", "judge", "workflow", "meta"]
    version: str
    status: Literal["candidate", "active", "pareto", "frozen", "retired"]
    instructions: str
    resources: Mapping[str, str]
    scope: SkillScope
    triggers: tuple[Trigger, ...]
    parent_ids: tuple[str, ...]
    safety_level: Literal["text_only", "review_required", "executable"]
```

```python
@dataclass(frozen=True)
class SkillUtility:
    skill_id: str
    uses: int
    successes: int
    marginal_utility: float
    mean_cost: float
    domain_utility: Mapping[str, float]
    temporal_utility: Mapping[str, float]
    negative_transfer_count: int
```

```python
@dataclass(frozen=True)
class RoutingDecision:
    selected_skill_ids: tuple[str, ...]
    rejected_skill_ids: tuple[str, ...]
    reasons: Mapping[str, str]
    confidence: float
    budget: RunBudget
    fallback_used: bool
```

```python
@dataclass(frozen=True)
class InferenceTrace:
    trace_id: str
    sample_public: Mapping[str, Any]
    routing: RoutingDecision
    specialist_reports: tuple[SpecialistReport, ...]
    aggregated_evidence: AggregatedEvidence
    decision: Prediction
    skill_versions: Mapping[str, str]
    usage: UsageRecord
    errors: tuple[str, ...]
```

```python
class ErrorType(StrEnum):
    ROUTING_MISS = ...
    EVIDENCE_MISS = ...
    EVIDENCE_HALLUCINATION = ...
    TEMPORAL_LEAKAGE = ...
    REASONING_ERROR = ...
    JUDGE_AGGREGATION_ERROR = ...
    LABEL_MAPPING_ERROR = ...
    ABSTENTION_ERROR = ...
```

```python
@dataclass(frozen=True)
class AttributionReport:
    trace_id: str
    error_types: tuple[ErrorType, ...]
    responsible_skill_ids: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]
    counterfactual_deltas: Mapping[str, float]
```

```python
class EvolutionOperation(StrEnum):
    ADD = ...
    EDIT = ...
    SPLIT = ...
    MERGE = ...
    GENERALIZE = ...
    SPECIALIZE = ...
    RETIRE = ...
    ROLLBACK = ...
```

```python
@dataclass(frozen=True)
class EvolutionProposal:
    proposal_id: str
    operation: EvolutionOperation
    target_skill_ids: tuple[str, ...]
    candidate_skills: tuple[SkillSpec, ...]
    source_trace_ids: tuple[str, ...]
    error_cluster_id: str | None
    rationale: str
    risk_flags: tuple[str, ...]
```

```python
@dataclass(frozen=True)
class EvaluationResult:
    per_sample: tuple[SampleEvaluation, ...]
    aggregate_metrics: Mapping[str, float]
    domain_metrics: Mapping[str, Mapping[str, float]]
    temporal_metrics: Mapping[str, Mapping[str, float]]
    confidence_intervals: Mapping[str, tuple[float, float]]
    usage: UsageRecord
```

```python
@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    disposition: Literal["active", "pareto", "rejected", "review_required"]
    baseline_result: EvaluationResult
    candidate_result: EvaluationResult
    paired_test: StatisticalTestResult
    regression_failures: tuple[str, ...]
    reason: str
```

### DEMSE 核心数据结构

```python
@dataclass(frozen=True)
class DomainEpisode:
    episode_id: str
    seed: int
    strategy: Literal["repeated_holdout", "leave_one_domain_out"]
    meta_train_domains: tuple[str, ...]
    meta_test_domains: tuple[str, ...]
    meta_train_sample_ids: tuple[str, ...]
    meta_test_sample_ids: tuple[str, ...]
    final_test_domains: tuple[str, ...]
    data_fingerprint: str
    skillbank_snapshot_id: str

@dataclass(frozen=True)
class CandidateIdentity:
    fingerprint: str
    operation: EvolutionOperation
    parent_skill_ids: tuple[str, ...]
    normalized_content_hash: str
    scope_signature: str

@dataclass(frozen=True)
class EpisodeEvaluation:
    episode_id: str
    candidate_fingerprint: str
    baseline_result: EvaluationResult
    candidate_result: EvaluationResult
    aggregate_deltas: Mapping[str, float]
    domain_deltas: Mapping[str, Mapping[str, float]]
    safety_level: str

@dataclass(frozen=True)
class TransferUtility:
    candidate_fingerprint: str
    episode_count: int
    tested_domains: tuple[str, ...]
    mean_gain: float
    std_gain: float
    confidence_interval: tuple[float, float]
    success_rate: float
    negative_transfer_rate: float
    worst_domain_drop: float
    coverage_delta: float
    calibration_delta: float
    cost_ratio: float
    domain_gains: Mapping[str, float]

@dataclass(frozen=True)
class MetaGateDecision:
    candidate_fingerprint: str
    accepted: bool
    disposition: Literal["generalized", "specialized", "pareto", "rejected", "retired", "review_required"]
    target_domains: tuple[str, ...]
    utility: TransferUtility
    failures: tuple[str, ...]
    reason: str
```

`MetaCheckpoint` 记录运行 ID、配置/数据指纹、源域、最终测试域、计划与已完成 episode、episode 结果和提交状态。所有对象采用不可变 dataclass，并用含 `schema_version` 的 JSON 持久化。

### DEMSE 核心接口

```python
class DomainEpisodeSampler:
    def build(self, samples, source_domains, final_test_domains, skillbank_snapshot_id) -> tuple[DomainEpisode, ...]: ...

class CandidateFirewall:
    def build_generation_view(self, episode, traces, attributions) -> MetaTrainView: ...
    def validate_candidate(self, candidate, episode) -> None: ...

class MetaEpisodeExecutor:
    async def execute(self, episode, baseline_skills) -> tuple[EpisodeEvaluation, ...]: ...

class CrossEpisodeAggregator:
    def aggregate(self, results) -> Mapping[str, TransferUtility]: ...

class MetaValidationGate:
    def decide(self, candidate, utility) -> MetaGateDecision: ...

class MetaEvolutionRunner:
    async def run(self, samples, *, final_test_domains, resume=False, evaluation_only=False) -> MetaEvolutionOutcome: ...
```

`MetaEpisodeExecutor` 与 `MetaValidationGate` 均无仓库写权限；最终提交由 `MetaEvolutionRunner` 在所有 episode 结束后统一执行。

## 核心接口

```python
class DatasetAdapter(Protocol):
    def discover(self, root: Path) -> DatasetDescriptor: ...
    def load(self, root: Path) -> Iterable[Sample]: ...
    def build_manifest(self, samples, policy, seed) -> DataManifest: ...
```

```python
class ModelBackend(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...
```

```python
class SkillRepository(Protocol):
    def list(self, status=None) -> list[SkillSpec]: ...
    def get(self, skill_id: str) -> SkillSpec: ...
    def stage(self, proposal: EvolutionProposal) -> tuple[SkillSpec, ...]: ...
    def promote(self, decision: GateDecision) -> None: ...
    def retire(self, skill_id: str, reason: str) -> None: ...
    def rollback(self, skill_id: str, version: str) -> None: ...
```

```python
class Router(Protocol):
    def select(
        self,
        sample: Mapping[str, Any],
        skills: Sequence[SkillSpec],
        utilities: Mapping[str, SkillUtility],
        budget: RunBudget,
    ) -> RoutingDecision: ...
```

```python
class AttributionEngine(Protocol):
    async def attribute(
        self,
        traces: Sequence[InferenceTrace],
        gold: Mapping[str, Any],
        replay: CounterfactualRunner | None,
    ) -> list[AttributionReport]: ...
```

```python
class EvolutionEngine(Protocol):
    async def propose(
        self,
        traces: Sequence[InferenceTrace],
        attributions: Sequence[AttributionReport],
        current_bank: Sequence[SkillSpec],
    ) -> list[EvolutionProposal]: ...
```

```python
class ValidationGate(Protocol):
    async def evaluate(
        self,
        proposal: EvolutionProposal,
        manifest: DataManifest,
        experiment: ExperimentConfig,
    ) -> GateDecision: ...
```

所有模型响应在进入领域对象前进行 Schema 校验；校验失败只形成错误记录，不改变 SkillBank。

## 模块设计

### `core`

定义领域模型、Schema 序列化和敏感字段过滤。仅依赖 Python 标准库与 Pydantic。

### `data`

实现四个 DatasetAdapter、数据指纹、去重、标签映射、时间解析、分层切分和泄漏检查。Weibo21、AMTCele 用于跨域实验；LiveFact 建立时间漂移实验；AdvFake 用作对抗鲁棒性测试。官方 test 或最终时间窗口只进入 test manifest。

### `skills`

加载、校验、查询和持久化 Skill；管理 active、candidate、pareto、frozen、retired 状态。版本仓库采用追加式事件日志与内容哈希快照，active 指针单独保存并使用原子替换。

Skill 包结构：

```text
skill/
├── SKILL.md
├── metadata.json
├── references/
├── templates/
├── assets/
└── scripts/
```

### `runtime`

构建 Coordinator、Specialist、Aggregator、Judge 链路，提供统一单样本和批量推理入口。支持确定性 `MockBackend` 和基于环境变量的 `OpenAICompatibleBackend`。推理阶段只能接收 `Sample.public_view()`；测试模式不会构造进化组件。

### `routing`

按适用范围、触发条件、相关性、历史边际效用、可靠性、成本和负迁移惩罚选择 Skill。随机路由、全专家和静态路由作为独立实验策略实现。

### `attribution`

执行错误分类、错误聚类与 Skill credit assignment。先运行规则级快速归因，再按预算执行替换、删除或新增单个 Skill 的反事实重放。

### `evolution`

蒸馏经验、产生生命周期操作、构造候选 Skill 和慢循环 Meta-Skill。Proposal 生成器没有直接写 active 仓库的权限。

### `validation`

执行脚本安全检查、配对重复推理、bootstrap 置信区间、McNemar/置换检验、多目标门控、保护域回归检测和 Pareto 分类。

默认晋升要求：全样本 Macro-F1 达到最小提升；coverage 不低于阈值；任一保护域退化不超过阈值；配对检验或置信区间满足配置；成本未超过预算；无安全或数据泄漏错误。

### `evaluation`

计算全样本指标、选择性预测指标、ECE/Brier、逐域/逐时间指标、成本和 Skill 演化统计。弃权样本在主分类指标中按未正确分类处理，同时另行报告已覆盖样本性能。

### `experiments`

组织 dry-run、evolve、validate、test、ablation 和 report，保存配置、manifest、checkpoint 和逐样本结果。八个正式实验臂共用同一 `ExperimentProtocol`。

### `security`

检查候选资源路径、AST、导入模块、危险调用、文件和网络权限，并生成审核报告。脚本执行采用独立临时目录和受限子进程，高风险脚本只能进入 `review_required`。

### `reporting`

从不可变实验记录生成 JSON、CSV 和 Markdown 报告，输出表格与绘图数据，显式标注无效、未完成或样本不足的运行。

### `cli`

提供稳定命令：

```text
evofact data inspect
evofact data manifest
evofact dry-run
evofact evolve
evofact validate
evofact test
evofact ablation
evofact skills list/show/diff/freeze/retire/rollback
evofact report
```

## 模块交互

训练与 Skill 快循环：

```text
DataRegistry
  → DataManifest(train/evolution-validation/protected-validation)
  → ExperimentRunner
  → InferenceRuntime
  → TraceStore
  → AttributionEngine
  → ExperienceDistiller
  → EvolutionEngine
  → Candidate SkillRepository
  → SecurityScanner
  → ValidationGate
  → promote / pareto / review-required / reject
  → UtilityTracker
  → 下一轮
```

最终测试路径为 Frozen DataManifest.test + Frozen Active SkillBank → InferenceRuntime → Evaluation → Reporting，不创建 EvolutionEngine，也不写 Skill 仓库。

Meta-Skill 慢循环从多轮 EvolutionEvent 中采样，评估归因、提案和门控的长期有效性，经独立验证后写入独立 Meta Skill 仓库。

## 文件组织

```text
EvoFactSkill/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── spec.md
├── plan.md
├── task.md
├── checklist.md
├── configs/
├── data/
├── skills/
├── src/evofact/
│   ├── cli.py
│   ├── config.py
│   ├── core/
│   ├── data/adapters/
│   ├── skills/
│   ├── runtime/
│   ├── routing/
│   ├── attribution/
│   ├── evolution/
│   ├── validation/
│   ├── evaluation/
│   ├── experiments/
│   ├── reporting/
│   └── security/
└── tests/
    ├── fixtures/
    ├── unit/
    ├── integration/
    └── e2e/
```

详细文件职责以 `task.md` 为准。

## 技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| Python 版本 | Python 3.11+ | 与现有环境兼容，具备现代类型能力 |
| 包结构 | `src` layout | 防止测试误导入工作目录源码 |
| 数据模型 | 标准库 frozen dataclass + 显式校验 | 无网络环境可直接运行，仍保持不可变与 Schema 明确 |
| 配置 | 受限 YAML 子集解析 + dataclass | 无第三方依赖，适合项目内可审计实验配置 |
| CLI | argparse | 标准库实现、无需安装即可执行 |
| 测试 | pytest | 支持 async 和 fixtures |
| 指标 | NumPy/scikit-learn + 自有选择性指标 | 成熟统计实现并保留研究指标可控性 |
| 统计 | paired bootstrap、McNemar、置换检验 | 适配分类结果和非正态效用差 |
| Skill 存储 | 内容哈希快照 + JSONL 事件日志 | 可审计、追加式、易回滚 |
| 候选选择 | 约束门控 + Pareto archive | 避免单指标和成本失控 |
| 模型接口 | OpenAI-compatible HTTP abstraction | 可接兼容服务，测试可替换 |
| 离线模式 | 确定性 MockBackend | 无网络仍能验收完整闭环 |
| 脚本安全 | AST denylist + 独立受限子进程 + 人审状态 | Python 无完美沙箱，采用分层防护 |
| 并发 | `asyncio`，每 worker 独立会话 | 适合模型 I/O并避免共享 Agent 状态 |
| 数据复制 | 不复制 CD-FND 数据 | 避免修改、重复分发和许可证风险，通过配置只读引用 |
| Meta-Skill | 默认关闭的独立慢循环 | 降低早期不稳定性并方便消融 |
| Git | EvoFactSkill 独立初始化 | 与 CD-FND 变更和历史彻底隔离 |

### DEMSE 增量文件

```text
src/evofact/
├── data/domains.py              # 域注册、数据指纹、最终测试域隔离
├── data/episodes.py             # repeated-holdout / leave-one-domain-out
├── evolution/identity.py        # 候选语义指纹
├── evolution/firewall.py        # meta-train 视图与泄漏阻断
├── evolution/meta.py            # 兼容名称下的 DEMSE 薄调度层
├── validation/transfer.py       # 跨 episode 迁移效用聚合
├── validation/meta_gate.py      # 外层元门控
├── experiments/meta_runner.py   # 双层 episode 执行器
├── experiments/checkpoint.py    # DEMSE 原子断点恢复
└── reporting/meta_report.py     # JSON / Markdown / CSV 报告

configs/demse_dry_run.yaml
tests/test_demse_models.py
tests/test_domain_episodes.py
tests/test_candidate_firewall.py
tests/test_transfer_aggregation.py
tests/test_meta_gate.py
tests/test_meta_checkpoint.py
tests/test_meta_runner.py
tests/test_cli_meta_evolve.py
```

### DEMSE 交互顺序

1. 加载配置、样本和 SkillBank 快照，隔离最终测试域。
2. 生成并验证全部域级 episode。
3. 仅在 meta-train 域执行推理、归因、蒸馏与候选生成。
4. 冻结 baseline/candidate，在 meta-test 域进行配对评估。
5. 原子写入 episode checkpoint，不修改 active SkillBank。
6. 按候选语义指纹聚合跨 episode 迁移效用。
7. 依次执行安全、证据量、统计、覆盖率、成本、负迁移和最差域约束。
8. 构造完整决策计划后集中原子提交，并输出论文实验报告。

### DEMSE 技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 元学习空间 | 离散 SkillBank，不更新模型参数 | 保持可解释、可审计并与论文主线一致 |
| 默认 episode | repeated holdout，每轮留一个 meta-test 域 | 兼顾领域轮换与可配置重复评估 |
| 严格复现模式 | leave-one-domain-out | 每个领域恰好作为一次模拟未知域 |
| 候选归并 | 规范化内容、操作、父 Skill、作用域的 SHA-256 | 跨轮稳定累计同类候选证据 |
| 配对公平性 | 相同样本顺序、后端种子、预算与重复次数 | 将差异归因于候选 SkillBank |
| 主效用 | `macro_f1_all` 增益 | 与现有主指标一致，弃权不被静默删除 |
| 约束 | coverage、ECE、cost ratio、negative transfer、worst-domain drop、安全 | 避免平均性能掩盖退化 |
| 统计支持 | 跨 episode bootstrap 置信区间下界 | 不用单次随机提升决定晋升 |
| 泛化门槛 | 默认至少 3 个有效 episode、95% CI 下界大于 0、负迁移率不高于 0.25 | 形成可配置的稳健默认值 |
| 特化 | 明确领域子集至少两次稳定增益后收窄 `SkillScope.domains` | 保留局部有效经验而避免全局负迁移 |
| 原子提交 | 全部 episode 完成后统一写仓库 | 防止评估过程污染 baseline |
| 恢复校验 | 配置、数据和 SkillBank 指纹必须同时一致 | 防止复用过期或不可比结果 |
| 兼容 | `meta_learning.enabled=false` 时保持原流程 | 不破坏已有命令与配置 |
| 离线验收 | 五领域 MockBackend 确定性场景 | 验证机制，不伪装成真实论文结果 |

## Spec 覆盖检查

- F1 由 `data` 与 DataManifest 覆盖。
- F2、F5 由 `runtime` 与 `routing` 覆盖。
- F3、F4、F11、F12、F17 由 `skills`、UtilityTracker 和 SkillRepository 覆盖。
- F6 由 `attribution` 覆盖。
- F7、F8、F18 由 `evolution` 覆盖。
- F9、F10、F13 由 `validation` 与 `security` 覆盖。
- F14、F15、F16 由 `experiments`、`evaluation`、`reporting` 和 `cli` 覆盖。
- F19、F20、F21 由 `data/domains.py` 与 `data/episodes.py` 覆盖。
- F22、F23、F27 由 `evolution/identity.py`、`evolution/firewall.py` 和原子提交协议覆盖。
- F24、F25、F26 由 `experiments/meta_runner.py`、`validation/transfer.py` 和 `validation/meta_gate.py` 覆盖。
- F28、F29、F30 由 DEMSE checkpoint、CLI、reporting 与消融配置覆盖。
