# adversarial-evolve 两折实验命令（Windows PowerShell）

在项目根目录执行。命令自动读取本地 `.env`，使用 `evofact.cmd`，无需重复设置密钥。推理与 optimizer 的模型设置沿用普通 evolve 两折配置；Generator 和 Verifier 使用推理后端。

最终测试使用 `outputs/static-100-before-pr-review/` 的同一批 `*-100.json` 样本及顺序：Fold 1 为 500 条，Fold 2 为 400 条。训练总量为 200 条、batch 为 16，每次跨域验证取 8 条真实新样本。

训练样本使用与普通 evolve 相同的 manifest 分区和领域均衡抽样，样本 ID 相同。随后组成混域 batch，两折均为 13 批：12 批各 16 条，最后一批 8 条。每批至少混合两个源领域，至少留出一个本批未出现的源领域供验证；允许验证已经训练过的样本，不固定留出一个验证领域。最终测试领域始终不参与训练或验证。

每个 batch 首次优化前生成一轮挑战，每轮最多 4 条。挑战通过独立核验后先推理，再将 trace 交给 optimizer。晋升失败后最多补充 2 轮挑战；达到上限后复用本批挑战及失败反馈继续优化，每批默认最多尝试 5 次。晋升成功后提交技能库、清空本批上下文并进入下一批。

**当前数据限制：本地 Weibo21 适配器不加载 `evidence`，目前加载的样本均没有证据。下面命令可启动这条路线，但挑战样本可能全部被证据核验器拒绝；接受数为 0 时，不能声称实验实现了有效的对抗样本增强。正式实验需先补充可追溯的真实证据并让数据加载器提供 `Sample.evidence`。真实标签不能代替证据，不要为凑数量编造证据。**

## Fold 1

政治、军事、科技、社会生活各 50 条，与普通 Fold 1 的训练 ID 相同。共 13 个混域 batch，每批混合 2–3 个源领域，从本批未出现的其他源领域验证。与 meta-evolve 的样本及 batch 划分一致。

训练：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_evolve_fold_1.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress adversarial-evolve --max-attempts-per-batch 5 --max-supplement-rounds 2 --output docs/weibo21_adversarial_evolve_fold_1/adversarial-evolve/train.json
```

训练完成后测试：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_evolve_fold_1.yaml --evolution-route adversarial-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_adversarial_evolve_fold_1/adversarial-evolve/final-test.json
```

## Fold 2

财经商业、教育考试、文体娱乐、医药健康、灾难事故各 40 条，与普通 Fold 2 的训练 ID 相同。共 13 个混域 batch，每批混合 2–4 个源领域，从本批未出现的其他源领域验证。与 meta-evolve 的样本及 batch 划分一致。

训练：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_evolve_fold_2.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress adversarial-evolve --max-attempts-per-batch 5 --max-supplement-rounds 2 --output docs/weibo21_adversarial_evolve_fold_2/adversarial-evolve/train.json
```

训练完成后测试：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_evolve_fold_2.yaml --evolution-route adversarial-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_adversarial_evolve_fold_2/adversarial-evolve/final-test.json
```

## 续跑

中断后，在原训练命令末尾加 `--resume`。训练自动读取配置中的 checkpoint，复用已经保存的生成响应、核验结果和挑战 trace。保持配置、训练数量、batch、并发、验证领域、补充生成上限和输出路径不变。

如果 `train.json` 中的状态为 `attempt_limit`，把原训练命令的 `--max-attempts-per-batch 5` 增大为 `10`，再加 `--resume`。如果状态为 `validation_pool_exhausted`，相同配置的验证池已经不足，重复续跑不会产生新样本。训练达到 `completed` 后，再运行最终测试。

测试中断时，在原 `test` 命令末尾加 `--resume`。已完成样本跳过，失败或缺失样本补跑。

**测试命令必须保留 `--evolution-route adversarial-evolve`，才能读取对抗路线的技能库。** 普通 evolve 的两折测试脚本仅针对普通路线，不要用它代替这里的测试命令。

## 中间结果位置

两折分别写入：

- `docs/weibo21_adversarial_evolve_fold_1/adversarial-evolve/`
- `docs/weibo21_adversarial_evolve_fold_2/adversarial-evolve/`

每个目录中：

- `training-traces.jsonl`：真实训练、挑战推理及基线、候选验证 trace。
- `train-checkpoint.json`：批次、尝试、候选、生成响应、核验结果、挑战 trace、失败反馈及预算。
- `train.json`：训练命令正常返回时的汇总；中途异常以 checkpoint 为准。
- `adversarial-batch-report.json`：训练决策及生成轮次报告；`accepted_challenge_samples` 为核验通过的挑战总数。
- `generation-audit.json`：正常返回后输出的生成与核验审计；单轮接受数在 `rounds[].verification.accepted` 中。
- `final-test.json`：持续更新的独立测试汇总，未完成时指标只覆盖已完成样本。
- `final-test.samples.jsonl`：逐条刷盘的测试结果、trace 和错误记录，续跑时不能删除。
- `final-test.checkpoint.json`：测试进度与结果文件路径。

技能库分别位于 `skills/store_weibo21_adversarial_evolve_fold_1/adversarial-evolve/` 和 `skills/store_weibo21_adversarial_evolve_fold_2/adversarial-evolve/`。首次空库从 `skills/seeds/` 开始；已有库则继续使用 active bank。两折相互独立，也不读取普通 evolve 或 meta-evolve 的演化结果。

`generation.batch_size` 控制每轮挑战数；`execution.batch_size` 控制真实训练批次；`data.evolution_validation_samples` 控制每轮真实验证数。这三者不是同一个 batch，也不是对同一条样本重复推理。

训练样本 ID、领域配额、批次数和批次大小与普通 evolve 一致，两折均为 13 批；为满足跨域验证，具体 batch 组合及顺序不同。`training_sampling` 保存样本 ID、配额和 batch 划分。旧的单领域或固定留出领域 checkpoint 不能恢复到新布局；跑过旧方案时，请为新实验使用新的技能库和输出目录，不覆盖旧结果。
