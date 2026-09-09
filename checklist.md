# EvoFactSkill Checklist

## 2026-09-09 单次 LLM 生成链路验收（当前版本）

- [x] 仅支持 proposer: llm；删除模板生成、策略权重与 --generation-mode。
- [x] 成功 trace 指导同类型升难度；失败 trace 指导同类型增样强化，数量由 LLM 自主选择。
- [x] 一次生成调用同时返回 samples 与独立 decisions，完整正文保留。
- [x] 结构、预算、重复、来源与留出泄漏校验；独立审核看不到生成标签，UNKNOWN/不一致拒绝。
- [x] construction/probe 按事件、证据文本和来源隔离；probe 与外层内容不进入生成请求。
- [x] 合格 samples 接原 evolve_once 和 DEMSE；没有合格生成时继续使用 construction 原样本。
- [x] 生成响应缓存支持审核后中断恢复；完成 episode 不重生成；正式审计提交幂等。
- [x] --samples 不强制 --facts；普通 mock 配置拒绝生产生成命令；测试通过注入后端离线运行。
- [x] 按 episode 导出可由 load_samples 读取的 JSONL，策略审计不写入 metadata。
- [x] 最终全套测试、语法与教程链接检查通过。

本版验收：完整 unittest discover 共 64 项通过，其中生成链路 16 项；包含五 episode 的单次生成、盲审核、字段/来源校验、JSONL 导出、evaluation-only、幂等提交、完成后恢复及生成后中断恢复。80 个源码/测试文件 AST 检查通过；learn、README 与设计文档共 13 份 Markdown 的 223 个本地链接全部有效；git diff --check 无空白错误。未调用真实 API，真实后端日期序列化使用 HTTP 替身验证；未据此宣称模型收益。Ruff 未安装，本轮仍未执行 lint。

边界：一次指生成调用，独立审核、检测和候选评估另有调用。上下文适应不等于生成模型参数训练；审核也不证明开放世界真实性。本版替代旧的数值模板三模式，旧 checkpoint/生成 bank 不兼容，使用新的 adversarial-llm 输出目录。

## 2026-09-08 复核（优先于下方历史勾选）

本轮按现有规格修复可复现性、路由、候选转换、恢复与提交问题，并扩充 learn 九章。下方 2026-09-06/07 的勾选保留作历史记录，不表示尚未实现的研究模块已经完成。

- [x] 安装入口统一到 main；回滚缺少快照时参数解析失败。
- [x] 空样本不回退到夹具；空技能库不回退到种子。
- [x] 所有路由策略按 domain/dataset/time 硬过滤作用域。
- [x] SPLIT/MERGE/RETIRE 的评估用技能表正确，冻结和重复候选被拒绝。
- [x] 固定验证在推理前检查训练/验证泄漏。
- [x] 指纹包含标签、证据、资源、触发器、模型和预算；恢复核对领域计划。
- [x] 提交后可恢复且不重复提交；指针写入失败不改变 active。
- [x] 配对数据必须完整唯一且标签一致；重复 episode 不增加证据。
- [x] 特化不绕过覆盖、成本、校准及最小 episode 数。
- [x] 退休依据为“移除操作”的正向收益，移除有害时拒绝。
- [x] learn 保留九章并增加可运行练习、测试依据、失败分析与实现限制。

验证结果（2026-09-08）：`python -m unittest discover -s tests -v` 共 48 项通过（原 31 项 + 新增 17 项回归）。70 个源码/测试 Python 文件 AST 解析通过；learn 共 10 份 Markdown、9 个 Day 章节、182 个本地链接检查通过。临时独立配置下 dry-run、validate、ablation、evolve、test、report、meta-evolve evaluation-only 及 resume 共 8 条 CLI 命令成功，输出均可解析为 JSON。运行产物使用临时目录隔离。

环境未安装 Ruff，本轮未执行 Ruff lint；AST 语法检查不等同于 lint。没有调用真实模型 API，也没有以 Mock 结果宣称真实数据效果。原有 `.gitignore` 对 `/learn` 的忽略规则保留，教程已写入本地但不会出现在默认 Git diff 中。

### 尚未完整落地的规格项

- [ ] F7/F8：成功经验驱动的多种结构操作自动发现、工具/工作流联合候选生成；当前自动提议主要为 ADD/EDIT。
- [ ] F14/F30：八个传统研究臂的独立算法实现与正式可比实验；部分臂目前共用策略。
- [ ] F18/AC18：慢循环策略的真实提案、独立门控与晋升；当前是关闭开关和事件记录接口。
- [ ] N8：真实模型 token/调用/并发预算全面执行及实际定价成本。
- [ ] F9/F15：重复观测依赖下的正式统计协议与真实多 seed 数据实验。
- [ ] N6：可执行脚本的外部受限沙箱；当前静态通过的脚本仍需审核，不自动执行。

## DEMSE 论文实现验收

### 需求与架构

- [x] DEMSE 被明确实现为离散 SkillBank 空间元学习，底层语言模型参数始终冻结。
- [x] 原固定验证闭环和已有 CLI 保持兼容。
- [x] 所有变更仅位于 EvoFactSkill。

### 领域与数据隔离

- [x] 所有参与 DEMSE 的样本具有有效领域标识。
- [x] meta-train 与 meta-test 以完整领域划分且没有交集。
- [x] repeated-holdout 和 leave-one-domain-out 均可运行。
- [x] 每个源领域至少担任一次 meta-test 域。
- [x] 最终测试域不出现在任何进化 episode 中。
- [x] 相同输入和随机种子生成相同 episode ID 与划分。

### 候选生成与防泄漏

- [x] 候选仅由 meta-train 脱敏轨迹、归因和蒸馏结果产生。
- [x] meta-test/final-test 的文本、标签和样本 ID 无法进入生成视图。
- [x] 泄漏扫描能拒绝包含禁用信息的候选。
- [x] 语义等价候选可跨 episode 归并，不同父 Skill、作用域或操作不会被错误归并。

### 配对评估与聚合

- [x] baseline/candidate 使用相同 meta-test 样本、顺序、种子和预算。
- [x] 每个 episode 输出 Macro-F1、coverage、ECE、selective risk、成本及逐域指标。
- [x] 正确计算平均增益、标准差、置信区间、成功率、负迁移率和最差域退化。
- [x] 有效 episode 数不足或仅单次随机提升时不允许全局激活。

### 元门控与生命周期

- [x] 稳定跨域增益候选进入 generalized/active。
- [x] 局部稳定增益候选进入 specialized 并收窄领域作用域。
- [x] 非支配候选进入 Pareto，显著负迁移候选被拒绝，持续有害 Skill 可产生 retired 决策。
- [x] 高风险候选进入 review_required，平均收益不能掩盖保护域严重退化。
- [x] 所有决策保留指标、失败约束和文本理由。

### 仓库与恢复

- [x] meta-test 期间 SkillRepository 只读，全部 episode 后才集中原子提交。
- [x] 提交失败不改变原 active SkillBank。
- [x] checkpoint 记录计划与已完成 episode，恢复时不重复执行。
- [x] 配置、数据或 SkillBank 指纹不匹配时拒绝恢复。

### CLI、实验、质量与文档

- [x] `evofact meta-evolve` 与 `configs/demse_dry_run.yaml` 可离线运行。
- [x] 五领域夹具及确定性场景覆盖泛化、特化、Pareto、负迁移和泄漏场景。
- [x] 固定门控对照与关键消融可运行。
- [x] JSON、Markdown、CSV 报告包含 episode 与聚合结果，Mock 结果被标记为机制测试。
- [x] 新增与原有测试全部通过，且不需要网络或 API Key。
- [x] `dry-run`、传统 `evolve`、`meta-evolve` 冒烟测试通过。
- [x] README 与论文术语、配置和命令一致，无密钥、缓存或临时输出被提交。
- [x] CD-FND 未因本任务产生变化。

> DEMSE 验收日期：2026-09-07。证据：31 项标准库测试通过；65 个包模块导入通过；repeated-holdout、leave-one-domain-out、checkpoint 恢复、原子提交、固定门控兼容与消融命令均完成验证。

> 验收日期：2026-09-06。证据：20 项标准库测试通过、59 个 Python 文件 AST 解析通过、离线源码构建通过、CLI dry-run/validate/ablation/report/evolve 实跑通过。

## 独立性与构建
- [x] 项目可独立构建和导入。
- [x] CD-FND 前后 Git 状态一致。
- [x] 语法扫描和完整测试通过。

## 数据与隔离
- [x] 四个 DatasetAdapter 均可注册、加载 fixture 和独立诊断。
- [x] 缺失数据不影响其他适配器。
- [x] Manifest 固定 seed 可复现。
- [x] 域、事件、重复内容和未来证据泄漏可检测。
- [x] AdvFake 和最终测试不会进入进化池。

## 推理与指标
- [x] dry-run 产生完整结构化 trace。
- [x] 推理输入中不存在 Gold 字段。
- [x] 弃权不能提高全样本主 Accuracy。
- [x] 输出 coverage、selective risk、ECE、Brier 和分组指标。
- [x] 测试模式不构造进化器或修改 SkillBank。

## SkillBank 与安全
- [x] 九个初始 Skill 全部可加载和路由。
- [x] 八种生命周期操作可持久化并重放。
- [x] 非法操作不改变 active 指针。
- [x] diff、父子关系、冻结、退休和回滚可查询。
- [x] 危险脚本被拒绝，脚本候选默认需要人工审核。
- [x] 持续负迁移产生降权或退休建议。

## 归因与进化
- [x] 八类错误归因具有测试场景。
- [x] 反事实重放生成 Skill 边际效用。
- [x] 经验可生成结构化 proposal。
- [x] Proposal 不含标签、答案或完整轨迹。
- [x] 结构生命周期操作不退化为文本 EDIT。
- [x] Meta-Skill 默认关闭且使用独立事件流。

## 验证与治理
- [x] baseline/candidate 使用独立验证样本与配对重复评估。
- [x] 单次随机提升无法通过正式门控。
- [x] coverage 显著下降的候选被拒绝。
- [x] 保护域严重退化的候选不进入 active。
- [x] 非支配候选可进入 Pareto archive。
- [x] 高风险候选进入 review_required。
- [x] 晋升或回滚失败时 active 不变。

## 实验与报告
- [x] 八实验臂共用运行器、Manifest 和指标。
- [x] checkpoint 恢复不重复完成样本。
- [x] 报告包含多 seed、CI、配对门控、校准、覆盖率、成本和分组结果。
- [x] 异常和不完整运行具有显式 run_status。
- [x] CLI 覆盖 data、dry-run、evolve、validate、test、ablation、skills 和 report。
- [x] README 快速验证无需网络和 API Key。

## 端到端
- [x] fixture → 推理 → 归因 → 反事实 credit → proposal → 独立验证 → Pareto/晋升决策 → 报告。
- [x] 保护域负迁移由门控阻止 active。
- [x] 危险脚本在推理前隔离。
- [x] Skill 晋升后可恢复旧版本。
- [x] 最终测试前后 seed SkillBank 内容不变。
