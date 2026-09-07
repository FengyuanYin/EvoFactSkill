# EvoFactSkill Tasks

> DEMSE 增量任务：以下 T-D1～T-D10 用于将论文中的域情景元门控技能进化落地。它们不替代原任务和固定验证基线，底层模型保持冻结。

## DEMSE 任务总览

| 任务 | 内容 | 依赖 |
|---|---|---|
| T-D1 | 配置与核心数据模型 | 无 |
| T-D2 | 域注册、最终测试隔离与 episode 采样 | T-D1 |
| T-D3 | 候选语义身份与信息防火墙 | T-D1 |
| T-D4 | 单 episode 内层进化与 meta-test 配对评估 | T-D2、T-D3 |
| T-D5 | 跨 episode 迁移效用聚合 | T-D4 |
| T-D6 | 外层元门控与状态决策 | T-D5 |
| T-D7 | checkpoint 与 SkillBank 原子提交 | T-D4、T-D6 |
| T-D8 | CLI、配置与报告 | T-D7 |
| T-D9 | 对照实验、消融与五领域夹具 | T-D8 |
| T-D10 | 文档、回归测试与最终验收 | T-D9 |

## T-D1：配置与领域模型

**步骤：** 在 `config.py` 增加 `MetaLearningConfig`；在 `core/models.py` 增加 `DomainEpisode`、`CandidateIdentity`、`EpisodeEvaluation`、`TransferUtility`、`MetaGateDecision`、`MetaCheckpoint` 与最终运行结果；实现构造期不变量和 JSON 序列化。

**验证：** 非法领域交集、非法指标、未知 disposition 和不完整 checkpoint 均被拒绝；旧配置仍可加载。

## T-D2：域注册与 Episode

**步骤：** 新建 `data/domains.py`、`data/episodes.py`；实现领域索引、数据指纹、最终测试域隔离、repeated-holdout、leave-one-domain-out 和 meta-test 全域轮换。

**验证：** 领域互斥、最终测试隔离、确定性、覆盖性及领域/样本不足错误测试通过。

## T-D3：候选身份与信息防火墙

**步骤：** 新建 `evolution/identity.py` 生成稳定候选语义指纹；新建 `evolution/firewall.py` 构造 meta-train-only 生成视图并阻断 meta-test/final-test 样本 ID、标签标记和内容泄漏。

**验证：** 等价候选能够归并，不同作用域或父 Skill 不会错误归并；泄漏候选被拒绝。

## T-D4：Episode 执行器

**步骤：** 新建 `experiments/meta_runner.py`；复用现有推理、归因、蒸馏和 proposal 逻辑；只在 meta-train 生成候选，在 meta-test 执行 baseline/candidate 配对评估；执行器不持有仓库写权限。

**验证：** 每个结果可追溯 episode 与候选指纹；配对双方使用相同样本、种子和预算；评估期间 active SkillBank 不变。

## T-D5：迁移效用聚合

**步骤：** 新建 `validation/transfer.py`；按候选语义身份聚合均值、标准差、bootstrap 置信区间、成功率、负迁移率、最差域退化、coverage、ECE 和成本。

**验证：** 正常、缺失、无效、领域覆盖不足和单 episode 输入均有明确且确定的结果。

## T-D6：Meta Gate

**步骤：** 新建 `validation/meta_gate.py`；实现 generalized、specialized、pareto、rejected、retired、review_required；特化决策收窄 `SkillScope.domains`。

**验证：** 最小 episode 数、置信区间、负迁移、保护域、coverage、成本和安全规则均有单元测试。

## T-D7：恢复与原子提交

**步骤：** 扩展 `experiments/checkpoint.py`；校验配置、数据和 SkillBank 指纹；跳过已完成 episode；先生成完整提交计划，再集中提交仓库。

**验证：** 中断恢复与连续运行等价；指纹不匹配拒绝恢复；任一前置失败不污染 active SkillBank。

## T-D8：CLI、配置与报告

**步骤：** 增加 `evofact meta-evolve`；支持 episode 数、采样策略、meta-test 域数、最终测试域、恢复和只评估参数；增加 `configs/demse_dry_run.yaml` 和 DEMSE JSON/Markdown/CSV 报告。

**验证：** 五领域离线命令可完成全流程，输出含领域轮换、逐 episode 指标、迁移效用、决策理由和版本变化。

## T-D9：实验与夹具

**步骤：** 构造不少于五个领域的确定性夹具；增加固定验证与 DEMSE 对照，以及无跨轮聚合、无最差域约束和无特化分支消融。

**验证：** 可控地覆盖泛化、特化、Pareto、负迁移拒绝和泄漏阻断，不把 Mock 指标表示为真实实验结果。

## T-D10：集成验收

**步骤：** 更新 README；运行原有与新增测试、`dry-run`、传统 `evolve`、`meta-evolve`；检查输出、缓存和项目边界。

**验证：** 全部验收项通过，所有改动仅位于 EvoFactSkill，CD-FND 未被本任务修改。

## 文件清单

| 操作 | 路径 | 职责 |
|---|---|---|
| 新建 | `pyproject.toml`, `.gitignore`, `README.md` | 项目元数据、依赖、文档 |
| 新建 | `configs/` | 默认、dry-run 与正式实验配置 |
| 新建 | `src/evofact/core/` | 不可变领域模型、Schema、脱敏 |
| 新建 | `src/evofact/data/` | 数据适配、Manifest、泄漏检查 |
| 新建 | `src/evofact/skills/` | Skill 加载、生命周期、版本库、效用 |
| 新建 | `src/evofact/runtime/` | 模型后端、推理链路、Trace |
| 新建 | `src/evofact/routing/` | Skill 选择与消融策略 |
| 新建 | `src/evofact/attribution/` | 错误归因、聚类、反事实重放 |
| 新建 | `src/evofact/evolution/` | 经验蒸馏、提案、生命周期操作、Meta 循环 |
| 新建 | `src/evofact/validation/` | 统计、多目标门控、Pareto |
| 新建 | `src/evofact/evaluation/` | 分类、选择性、校准、分组指标 |
| 新建 | `src/evofact/experiments/` | 实验协议、运行器、checkpoint |
| 新建 | `src/evofact/reporting/` | JSON/CSV/Markdown 报告 |
| 新建 | `src/evofact/security/` | 候选脚本安全扫描 |
| 新建 | `src/evofact/cli.py` | 统一命令行入口 |
| 新建 | `skills/seeds/` | 初始可审计事实核验 Skill |
| 新建 | `tests/` | 单元、集成和端到端测试 |

## T1：初始化独立项目

**依赖：** 无  
**步骤：** 建立 src-layout 包、依赖、测试和静态检查配置；配置忽略密钥、缓存、数据和输出；初始化独立 Git 仓库。  
**验证：** 安装本地包并导入 `evofact`；记录 CD-FND 前后 Git 状态并确认一致。

## T2：配置模型与 YAML 加载

**依赖：** T1  
**步骤：** 定义数据路径、模型、预算、切分、验证门槛、Meta 开关和输出位置；支持 YAML、环境变量覆盖和配置快照。  
**验证：** 默认配置、dry-run 配置和非法配置测试通过。

## T3：Sample 与 Evidence 模型

**依赖：** T1  
**步骤：** 定义样本、证据、时间和元数据模型；规范化标签与 ID。  
**验证：** JSON 序列化往返、冻结属性和非法输入测试通过。

## T4：Skill、Trace、Proposal 与 Gate 模型

**依赖：** T3  
**步骤：** 定义 Plan 中的 SkillSpec、SkillUtility、RoutingDecision、InferenceTrace、AttributionReport、EvolutionProposal、EvaluationResult 和 GateDecision。  
**验证：** 枚举、状态约束、父版本与 Schema 版本测试通过。

## T5：敏感字段过滤

**依赖：** T3  
**步骤：** 递归删除 label、gold、answer、target、expected 等字段；建立测试标签隔离标记。  
**验证：** 嵌套字典、列表和大小写变体均无法泄漏标签。

## T6：DatasetAdapter 与 DataRegistry

**依赖：** T3  
**步骤：** 定义 discover/load 接口、诊断对象、适配器注册和可用性枚举。  
**验证：** 四个适配器均可发现；缺失数据返回明确诊断。

## T7：Weibo21 适配器

**依赖：** T6  
**步骤：** 读取 JSONL/JSON/PKL 可用格式，映射 content/label/category，过滤未知域并记录统计。  
**验证：** fixture 的数量、标签和中文域映射正确。

## T8：AMTCele 适配器

**依赖：** T6  
**步骤：** 读取 CSV/JSONL，规范化 legit/fake 标签和带数字后缀的域名。  
**验证：** 七域映射和无效行诊断测试通过。

## T9：LiveFact 适配器

**依赖：** T6  
**步骤：** 发现月份目录与 cls/inf、-3/0/+3 文件，解析时间窗口和证据快照。  
**验证：** 月份、相对时间和任务类型 fixture 正确。

## T10：AdvFake 适配器

**依赖：** T6  
**步骤：** 读取 CSV，构造原始/对抗文本关联和攻击类型元数据，将其标为 robustness-only。  
**验证：** 样本配对、标签和隔离角色测试通过。

## T11：指纹、去重和 DataManifest

**依赖：** T7–T10  
**步骤：** 对规范化样本生成稳定指纹；检测跨 split 重复；按域、事件和时间生成四区 Manifest。  
**验证：** 同 seed 与输入生成同 manifest_id；顺序变化不改变指纹。

## T12：泄漏检测

**依赖：** T11  
**步骤：** 检测样本 ID、内容指纹、事件和非法时间穿越；限制 AdvFake 与最终测试用途。  
**验证：** 人造跨域、跨事件、重复内容和未来证据泄漏全部被报告。

## T13：全样本和选择性指标

**依赖：** T3  
**步骤：** 实现 accuracy_all、macro_f1_all、coverage、covered_accuracy、selective_risk、risk-coverage curve；弃权按未正确分类计入主指标。  
**验证：** 1/50 决策不能得到主 Accuracy=1.0。

## T14：校准和分组指标

**依赖：** T13  
**步骤：** 实现 Brier、ECE、逐域、逐时间、最差域与成本聚合。  
**验证：** 手工可计算 fixture 与实现结果一致。

## T15：配对统计

**依赖：** T13  
**步骤：** 实现 paired bootstrap CI、McNemar 与配对置换检验；所有随机过程接收 seed。  
**验证：** 固定输入结果可重复，明显优劣样本能被识别。

## T16：多目标效用与 Pareto

**依赖：** T14  
**步骤：** 定义性能、覆盖、校准、成本、延迟和最差域目标；实现约束检查、支配和非支配归档。  
**验证：** 支配、互不支配和违反硬约束的候选分类正确。

## T17：Skill 包加载

**依赖：** T4  
**步骤：** 解析 SKILL.md frontmatter、metadata 和资源；约束路径、大小、类型和名称；兼容 CD-FND 基础格式。  
**验证：** 合法包加载，越界路径、重复名和非法状态被拒绝。

## T18：初始 SkillBank

**依赖：** T17  
**步骤：** 创建 routing、claim decomposition、evidence assessment、temporal reasoning、source credibility、linguistic manipulation、numerical consistency、cross-source contradiction 和 judge seed Skill。  
**验证：** 全部 seed 包通过加载与 Schema 校验。

## T19：版本仓库

**依赖：** T17  
**步骤：** 实现内容哈希快照、追加事件日志、active 指针、原子写入、diff 与事件重放。  
**验证：** 写入失败不改变 active；历史能够重建当前状态。

## T20：八种生命周期操作

**依赖：** T19  
**步骤：** 实现 ADD、EDIT、SPLIT、MERGE、GENERALIZE、SPECIALIZE、RETIRE、ROLLBACK 的前置条件与事件。  
**验证：** 每个操作至少覆盖成功、非法输入和原子失败路径。

## T21：SkillUtility 与负迁移追踪

**依赖：** T19  
**步骤：** 聚合使用、成功、反事实边际效用、成本、逐域和逐时间效用；生成降权/退休建议。  
**验证：** 持续有害 Skill 达到阈值，短期抖动不会误淘汰。

## T22：脚本安全扫描

**依赖：** T17  
**步骤：** 检查语法、AST、import、危险调用、路径和公开函数；生成安全级别与审核原因。  
**验证：** 文件删除、进程、网络、动态执行等危险 fixture 被阻止。

## T23：模型后端

**依赖：** T2  
**步骤：** 定义请求/响应/usage 接口；实现确定性 MockBackend 和安全读取凭据的 OpenAI-compatible 后端。  
**验证：** Mock 多次结果一致；无凭据不泄漏且给出明确错误。

## T24：Skill Router 与实验策略

**依赖：** T18、T21  
**步骤：** 实现效用感知评分、预算截断和安全兜底；实现 static、all-experts、random、utility-aware 策略。  
**验证：** retired 不可选、frozen 可选、预算有效、随机策略由 seed 控制。

## T25：推理链路与 TraceStore

**依赖：** T5、T23、T24  
**步骤：** 实现 Coordinator、并发 Specialist、确定性聚合和 Judge；记录 Skill 版本、usage、错误和不可变 trace。  
**验证：** MockBackend 离线运行得到完整 trace；测试路径不写 Skill 仓库。

## T26：规则归因与错误聚类

**依赖：** T25  
**步骤：** 实现八类错误规则、置信度和证据；按错误类型、域、触发器和报告特征稳定聚类。  
**验证：** 每类错误 fixture 命中预期，聚类由 seed 控制。

## T27：反事实贡献估计

**依赖：** T24–T26  
**步骤：** 实现 leave-one-skill-out 与替换重放，缓存共同结果，输出配对效用 delta。  
**验证：** 人造有益/有害 Skill 得到正负方向正确的 delta。

## T28：经验蒸馏与提案

**依赖：** T20、T26  
**步骤：** 选择稳定成功模式和错误簇；去标签、去样本化；生成结构化生命周期提案和风险标记。  
**验证：** 输出不包含 gold、答案和整段轨迹；EDIT 与结构操作选择正确。

## T29：联合进化与 Meta 慢循环

**依赖：** T28  
**步骤：** 支持多 Skill/路由/工作流联合 proposal；实现独立 Meta 配置、事件、仓库和周期门槛。  
**验证：** 默认不运行 Meta；启用后不修改普通 Skill 事件流。

## T30：候选评估与晋升门控

**依赖：** T12、T15、T16、T22、T25  
**步骤：** 对 baseline/candidate 运行相同样本、seed 和重复次数；检查统计、最小提升、覆盖、成本、安全和保护域退化；分类为 active、pareto、review_required 或 rejected。  
**验证：** 单次随机提升、严重弃权、保护域退化和危险脚本均不能自动晋升。

## T31：实验协议、断点、消融与报告

**依赖：** T25–T30  
**步骤：** 实现 dry-run/evolve/validate/test、checkpoint、八实验臂、统一 JSON/CSV/Markdown 输出和演化曲线数据。  
**验证：** 中断恢复不重复完成样本；所有实验臂共享 manifest 与指标定义。

## T32：CLI、文档和完整验收测试

**依赖：** T1–T31  
**步骤：** 接入全部 CLI 子命令；完成 README、数据说明、复现指南；增加单元、集成、E2E 与独立性检查。  
**验证：** 无网络、无 API Key 执行静态检查、完整测试和 dry-run 均通过；CD-FND Git 状态与开发前快照一致。

## 执行顺序

```text
T1 → T2–T5
   → T6 → T7–T10 → T11 → T12
   → T13 → T14–T16
   → T17 → T18–T22
   → T23 → T24 → T25
   → T26 → T27–T29
   → T30 → T31 → T32
```

T31 前只使用离线 fixture 和 MockBackend，不调用真实模型 API。
