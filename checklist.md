# EvoFactSkill Checklist

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
