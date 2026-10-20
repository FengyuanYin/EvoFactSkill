# meta-evolve 两折实验命令（Windows PowerShell）

新启动实验支持 optimizer 自主提出多技能联合修改，整组验证和晋升；旧 checkpoint 续跑保持单技能模式，详见 [联合优化说明](optimizer持久记忆与上下文管理.md#多技能联合优化)。

两折配置已启用跨批 optimizer 记忆和 token 窗口管理，详见 [Optimizer 持久记忆教程](optimizer持久记忆与上下文管理.md)。按当前实验使用的 DeepSeek V4 Flash 窗口设置 `context_window: 1000000`（tokens）；更换 optimizer 后端模型时同步调整。此配置更新不能直接续跑旧 checkpoint；新训练需使用新的保存路径，旧实验结果保留。

在项目根目录执行。命令自动读取本地 `.env`，使用 `evofact.cmd`，无需重复设置密钥。推理与 optimizer 的模型设置沿用普通 evolve 两折配置。

最终测试使用 `outputs/static-100-before-pr-review/` 的同一批 `*-100.json` 样本及顺序：Fold 1 为 500 条，Fold 2 为 400 条。训练总量为 200 条、batch 为 16，每次验证取 8 条真实新样本。

meta-evolve 复用普通 evolve 的 manifest 训练分区、领域均衡抽样和相同 200 个训练样本，再组成混域 batch。两折均为 13 批：12 批各 16 条，最后一批 8 条。每批至少混合两个源领域，至少留出一个本批未出现的源领域供验证；允许验证训练过的样本，不固定留出一个领域。最终测试领域不参与训练或验证。

每批直接优化，晋升后立即进入下一批；失败则累积验证 trace，换尚未用于验证的样本继续优化，默认每批最多尝试 5 次。晋升后清空本批上下文，保留更新后的技能库。

可选的新实验方案见[独立均衡验证与 FAKE 召回保护](独立均衡验证实验启动教程-Windows.md)：保持每批跨域验证范式，使用全新配置、时间标签和技能库。本教程原配置及旧断点继续保留原策略。

## Fold 1

训练领域与普通 Fold 1 一致：政治、军事、科技、社会生活各 50 条，相同样本 ID。共 13 个混域 batch，每批混合 2–3 个源领域。例如本批有政治、军事、科技时，从社会生活中取验证样本；本批有政治、军事时，从科技、社会生活中取验证样本。最终测试保持原来的五个领域、500 条。

训练：

```powershell
.\evofact.cmd --config configs/weibo21_meta_evolve_fold_1.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress meta-evolve --max-attempts-per-batch 5 --output docs/weibo21_meta_evolve_fold_1/meta-evolve/train.json
```

训练完成后测试：

```powershell
.\evofact.cmd --config configs/weibo21_meta_evolve_fold_1.yaml --evolution-route meta-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_meta_evolve_fold_1/meta-evolve/final-test.json
```

## Fold 2

训练领域与普通 Fold 2 一致：财经商业、教育考试、文体娱乐、医药健康、灾难事故各 40 条，相同样本 ID。共 13 个混域 batch，每批混合 2–4 个源领域，从本批未出现的其他源领域取验证样本。最终测试保持原来的四个领域、400 条。

训练：

```powershell
.\evofact.cmd --config configs/weibo21_meta_evolve_fold_2.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress meta-evolve --max-attempts-per-batch 5 --output docs/weibo21_meta_evolve_fold_2/meta-evolve/train.json
```

训练完成后测试：

```powershell
.\evofact.cmd --config configs/weibo21_meta_evolve_fold_2.yaml --evolution-route meta-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_meta_evolve_fold_2/meta-evolve/final-test.json
```

## 每领域 25 条和 50 条 final-test

以下命令在训练完成后执行，使用原来的已训练技能库。每个领域固定读取原 `*-100.json` 的前 25 条或前 50 条，与从已有 100 条结果截取的样本一致。`--limit` 只控制训练数量，不能用来指定测试条数。

先生成独立测试配置，保留原配置及全部训练路径，只修改每领域测试数量。在项目根目录执行一次：

```powershell
$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($fold in 1, 2) {
    $source = "configs/weibo21_meta_evolve_fold_$fold.yaml"
    $text = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $source), $utf8)
    $pattern = '(?m)^([ \t]*final_test_samples_per_domain:[ \t]*)100[ \t]*\r?$'
    if ([regex]::Matches($text, $pattern).Count -ne 1) { throw "Expected exactly one 100/domain setting: $source" }
    $targetRoot = "docs/weibo21_meta_evolve_fold_$fold/meta-evolve"
    New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
    foreach ($n in 25, 50) {
        $config = [regex]::Replace($text, $pattern, ('${1}' + $n))
        [System.IO.File]::WriteAllText("$targetRoot/config.final-test-$n.yaml", $config, $utf8)
    }
}
```

测试命令须保留 `--evolution-route meta-evolve`，读取本路线已训练的技能库。

### final-test 25（每领域 25 条）

Fold 1 共 125 条，Fold 2 共 100 条，两折共 225 条：

```powershell
.\evofact.cmd --config docs/weibo21_meta_evolve_fold_1/meta-evolve/config.final-test-25.yaml --evolution-route meta-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_meta_evolve_fold_1/meta-evolve/final-test-25.json
.\evofact.cmd --config docs/weibo21_meta_evolve_fold_2/meta-evolve/config.final-test-25.yaml --evolution-route meta-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_meta_evolve_fold_2/meta-evolve/final-test-25.json
```

### final-test 50（每领域 50 条）

Fold 1 共 250 条，Fold 2 共 200 条，两折共 450 条：

```powershell
.\evofact.cmd --config docs/weibo21_meta_evolve_fold_1/meta-evolve/config.final-test-50.yaml --evolution-route meta-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_meta_evolve_fold_1/meta-evolve/final-test-50.json
.\evofact.cmd --config docs/weibo21_meta_evolve_fold_2/meta-evolve/config.final-test-50.yaml --evolution-route meta-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_meta_evolve_fold_2/meta-evolve/final-test-50.json
```

每条测试命令都会调用模型重新推理。已有完整 100 条结果且只需统计子集时，可使用已导出的 25／50 条结果，无需再启动测试。25 条和 50 条分别生成 `final-test-25.*`、`final-test-50.*` 汇总、逐样本记录及测试断点，不覆盖原 `final-test.*` 或训练结果。测试中断后，在对应命令末尾添加 `--resume`，保持配置、模型和技能库不变；不要将不同测试规模的记录或断点混用。


## 续跑

中断后，在原训练命令末尾加 `--resume`；无需传 `--checkpoint`，训练自动使用配置中的 `meta_learning.checkpoint_path`。保持配置、训练数量、batch、并发、验证领域和输出路径不变。

当前 batch 达到尝试上限后，保留当前技能库，在 `skipped_batches` 记录 `reason=attempt_limit`，自动进入下一批，不强制晋升。旧版本停在 `status=attempt_limit` 的断点可直接使用原命令加 `--resume`，无需提高上限。`completed_batches` 包含无更新但已处理的批次。如果状态为 `validation_pool_exhausted`，表示没有足够的新验证样本，继续执行相同命令不会补充样本池。训练状态达到 `completed` 后，再运行最终测试。

测试中断时，在原 `test` 命令末尾加 `--resume`。已完成样本跳过，失败或缺失样本补跑；默认每条超时或网络错误重试两次。

**测试命令必须保留 `--evolution-route meta-evolve`，才能读取这条路线的技能库。**

## 中间结果位置

两折分别写入：

- `docs/weibo21_meta_evolve_fold_1/meta-evolve/`
- `docs/weibo21_meta_evolve_fold_2/meta-evolve/`

每个目录中：

- `training-traces.jsonl`：训练与基线、候选验证的推理 trace。
- `train-checkpoint.json`：逐批训练状态、失败反馈、候选、已用验证样本、技能库及预算。
- `train.json`：训练命令正常返回时的汇总；中途异常以 checkpoint 为准。
- `meta-batch-report.json`：每批尝试及晋升决策报告。
- `final-test.json`：持续更新的独立测试汇总，未完成时指标只覆盖已完成样本。
- `final-test.samples.jsonl`：逐条刷盘的测试结果、trace 和错误记录，续跑时不能删除。
- `final-test.checkpoint.json`：测试进度与结果文件路径。

技能库分别位于 `skills/store_weibo21_meta_evolve_fold_1/meta-evolve/` 和 `skills/store_weibo21_meta_evolve_fold_2/meta-evolve/`。首次空库从 `skills/seeds/` 开始；已有库则继续使用 active bank。两折相互独立，也不读取普通 evolve 或 adversarial-evolve 的演化结果。

训练领域、配额、样本 ID、批次数和批次大小与普通 evolve 对齐，两折均为 13 批。为满足跨域验证，具体 batch 组合及顺序与普通 evolve 不同。`train.json` 中 `training_sampling` 记录原始训练样本顺序、各域数量及实际 batch 划分。旧的单领域或固定留出领域 checkpoint 与新布局不兼容，不能直接续用；已经跑过旧方案时，需为新实验使用新的技能库和输出目录，不覆盖旧结果。
