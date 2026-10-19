# 普通 evolve 两折实验命令（Windows PowerShell）

在项目根目录执行。命令自动读取本地 `.env`，无需激活环境、安装命令入口或重复填写密钥。`.env` 和 `docs/` 下的实验产物由 Git 忽略，三个启动教程纳入版本管理。

最终测试固定读取 `outputs/static-100-before-pr-review/` 中对应领域的 `*-100.json`，保持其中的样本 ID 和顺序。Fold 1 测试 500 条，Fold 2 测试 400 条。

两折依次测试、自动识别续跑（可以反复执行，已完成样本不会重复调用模型）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/evaluate_evolve_folds.ps1
```

单折失败会保留结果并继续测试另一折，脚本最后报告未完成的折。保持电脑通电、不睡眠。首次运行无需 `--resume`；以后单独续跑某折时，在下方 `test` 命令末尾加 `--resume`。不要更换模型、技能库或测试样本顺序。每条超时或网络错误默认重试两次，仍失败就记录错误并继续后面的样本；下一次续跑只补失败和未完成的样本。

## Fold 1

训练：

```powershell
.\evofact.cmd --config configs/weibo21_evolve_fold_1.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress evolve --checkpoint docs/weibo21_evolve_fold_1/evolve-checkpoint.json --output docs/weibo21_evolve_fold_1/evolve.json
```

训练完成后测试：

```powershell
.\evofact.cmd --config configs/weibo21_evolve_fold_1.yaml --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_evolve_fold_1/final-test.json
```

## Fold 2

训练：

```powershell
.\evofact.cmd --config configs/weibo21_evolve_fold_2.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress evolve --checkpoint docs/weibo21_evolve_fold_2/evolve-checkpoint.json --output docs/weibo21_evolve_fold_2/evolve.json
```

训练完成后测试：

```powershell
.\evofact.cmd --config configs/weibo21_evolve_fold_2.yaml --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_evolve_fold_2/final-test.json
```

## 每领域 25 条和 50 条 final-test

以下命令在训练完成后执行，使用原来的已训练技能库。每个领域固定读取原 `*-100.json` 的前 25 条或前 50 条，与从已有 100 条结果截取的样本一致。`--limit` 只控制训练数量，不能用来指定测试条数。

先生成独立测试配置，保留原配置及全部训练路径，只修改每领域测试数量。在项目根目录执行一次：

```powershell
$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($fold in 1, 2) {
    $source = "configs/weibo21_evolve_fold_$fold.yaml"
    $text = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $source), $utf8)
    $pattern = '(?m)^([ \t]*final_test_samples_per_domain:[ \t]*)100[ \t]*\r?$'
    if ([regex]::Matches($text, $pattern).Count -ne 1) { throw "Expected exactly one 100/domain setting: $source" }
    $targetRoot = "docs/weibo21_evolve_fold_$fold"
    New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
    foreach ($n in 25, 50) {
        $config = [regex]::Replace($text, $pattern, ('${1}' + $n))
        [System.IO.File]::WriteAllText("$targetRoot/config.final-test-$n.yaml", $config, $utf8)
    }
}
```

### final-test 25（每领域 25 条）

Fold 1 共 125 条，Fold 2 共 100 条，两折共 225 条：

```powershell
.\evofact.cmd --config docs/weibo21_evolve_fold_1/config.final-test-25.yaml --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_evolve_fold_1/final-test-25.json
.\evofact.cmd --config docs/weibo21_evolve_fold_2/config.final-test-25.yaml --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_evolve_fold_2/final-test-25.json
```

### final-test 50（每领域 50 条）

Fold 1 共 250 条，Fold 2 共 200 条，两折共 450 条：

```powershell
.\evofact.cmd --config docs/weibo21_evolve_fold_1/config.final-test-50.yaml --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_evolve_fold_1/final-test-50.json
.\evofact.cmd --config docs/weibo21_evolve_fold_2/config.final-test-50.yaml --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_evolve_fold_2/final-test-50.json
```

每条测试命令都会调用模型重新推理。已有完整 100 条结果且只需统计子集时，可使用已导出的 25／50 条结果，无需再启动测试。25 条和 50 条分别生成 `final-test-25.*`、`final-test-50.*` 汇总、逐样本记录及测试断点，不覆盖原 `final-test.*` 或训练结果。测试中断后，在对应命令末尾添加 `--resume`，保持配置、模型和技能库不变；不要将不同测试规模的记录或断点混用。

## 中间结果位置

两折分别写入 `docs/weibo21_evolve_fold_1/` 和 `docs/weibo21_evolve_fold_2/`：

- `training-traces.jsonl`：训练、基线验证和候选验证的完整推理轨迹。
- `evolve-checkpoint.json`：逐批候选、门控决策、晋升记录、技能库身份及预算状态。
- `validation-curve.csv`、`validation-active.csv`、`validation-curve.svg`：验证曲线产物。
- `evolve.json`：训练完成后的汇总。
- `final-test.json`：持续更新的汇总、指标和使用的技能库身份，不内嵌所有 trace；未完成时指标只覆盖已完成样本。
- `final-test.samples.jsonl`：立即刷盘的逐样本 trace、评估结果、错误记录及预算增量，作为续跑依据，不能删除。
- `final-test.checkpoint.json`：测试状态、完成数量和结果文件路径；可由逐条文件恢复。

技能包分别保存在 `skills/store_weibo21_evolve_fold_1/` 和 `skills/store_weibo21_evolve_fold_2/`，两折不共享技能库。上述训练命令会继续使用各自已有的 active store；首次空 store 自动从 `skills/seeds/` 开始。
