# adversarial-evolve 两折实验命令（Windows PowerShell）

可选的 Generator 协议/标签经验、版本化能力边界、覆盖记忆和 token 管理见 [Adversarial Generator 记忆教程](adversarial-generator记忆与上下文管理.md)，其中提供同时启用 optimizer 与 generator 记忆的新两折配置。旧实验不自动切换。

可选的跨批 optimizer 记忆、token 窗口管理及新实验配置见 [Optimizer 持久记忆教程](optimizer持久记忆与上下文管理.md)。默认配置和已启动实验保持原行为。

在项目根目录执行。命令自动读取本地 `.env`，使用 `evofact.cmd`，无需重复设置密钥。推理与 optimizer 的模型设置沿用普通 evolve 两折配置；Generator 和 Verifier 使用推理后端。

优化器上下文已取消单字符串 8,000、技能内容 60,000 和整体 300,000 字符的截断与裁剪。原 `evolution.max_*_chars` 配置值保留兼容，不再用于裁剪优化器输入；生成器请求仍有独立限制。服务端模型上下文窗口仍适用。

最终测试使用 `outputs/static-100-before-pr-review/` 的同一批 `*-100.json` 样本及顺序：Fold 1 为 500 条，Fold 2 为 400 条。训练总量为 200 条、batch 为 16，每次跨域验证取 8 条真实新样本。

训练样本使用与普通 evolve 相同的 manifest 分区和领域均衡抽样，样本 ID 相同。随后组成混域 batch，两折均为 13 批：12 批各 16 条，最后一批 8 条。每批至少混合两个源领域，至少留出一个本批未出现的源领域供验证；允许验证已经训练过的样本，不固定留出一个验证领域。最终测试领域始终不参与训练或验证。

每个 batch 首次优化前生成一轮挑战，每轮最多 4 条。挑战通过独立核验后先推理，再将 trace 交给 optimizer。晋升失败后最多补充 2 轮挑战；达到上限后复用本批挑战及失败反馈继续优化，每批默认最多尝试 5 次。晋升成功后提交技能库、清空本批上下文并进入下一批。

生成器若超量返回，只有在样本与决策 ID 唯一且完整一一配对时，才保留前 4 条候选进入完整核验；超出条目记录为拒绝，原响应保留用于审计。数量不匹配、重复决策或顶层结构错误等响应整轮拒绝，不因此中断训练；优化器继续使用真实训练 trace 和本批其他有效挑战。已缓存的无效响应在续跑时也按此处理，无需删除断点。

对抗生成基于训练文本、训练真实标签、模型内源知识及已消耗的验证反馈，不要求外部 `evidence`。Weibo21 的空证据字段按原样保留，不会因为为空而拒绝样本。生成器根据训练 trace 选择保持标签的改写或改变事实与目标标签的挑战；标签变化的理由保存在审计记录，不进入样本正文。

核验器使用源训练文本及其真实标签、模型自身知识独立判断生成文本，不接收生成器给出的目标标签或理由；判断一致才接受，无法确定则拒绝。这里是模型知识与标签一致性复核，不是外部证据核查，也不保证生成标签绝对正确。最终晋升仍由真实验证样本决定，验证与最终测试样本不能用于构造挑战。

**本教程已切换为 `endogenous-knowledge-v1`。** 新配置使用 `weibo21_adversarial_knowledge_fold_*` 的独立保存目录和技能库，训练数量、领域、模型及 batch 参数沿用原配置。原 `weibo21_adversarial_evolve_fold_*` 的证据模式训练记录全部保留；旧断点不能续跑到新策略。若旧进程仍在运行，请先在其终端按 Ctrl+C 保存退出，再按本教程新命令启动。新策略后续可以正常 `--resume`。

可选的新实验方案见[独立均衡验证与 FAKE 召回保护](独立均衡验证实验启动教程-Windows.md)：保持每批跨域验证范式，独立验证样本不进入生成与优化反馈，使用全新配置和保存目录。本教程原配置及旧断点继续保留原策略。

## Fold 1

政治、军事、科技、社会生活各 50 条，与普通 Fold 1 的训练 ID 相同。共 13 个混域 batch，每批混合 2–3 个源领域，从本批未出现的其他源领域验证。与 meta-evolve 的样本及 batch 划分一致。

训练：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_knowledge_fold_1.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress adversarial-evolve --max-attempts-per-batch 5 --max-supplement-rounds 2 --output docs/weibo21_adversarial_knowledge_fold_1/adversarial-evolve/train.json
```

训练完成后测试：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_knowledge_fold_1.yaml --evolution-route adversarial-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_adversarial_knowledge_fold_1/adversarial-evolve/final-test.json
```

## Fold 2

财经商业、教育考试、文体娱乐、医药健康、灾难事故各 40 条，与普通 Fold 2 的训练 ID 相同。共 13 个混域 batch，每批混合 2–4 个源领域，从本批未出现的其他源领域验证。与 meta-evolve 的样本及 batch 划分一致。

训练：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_knowledge_fold_2.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress adversarial-evolve --max-attempts-per-batch 5 --max-supplement-rounds 2 --output docs/weibo21_adversarial_knowledge_fold_2/adversarial-evolve/train.json
```

训练完成后测试：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_knowledge_fold_2.yaml --evolution-route adversarial-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_adversarial_knowledge_fold_2/adversarial-evolve/final-test.json
```

## 自动依次训练两个 Fold（自动识别断点）

在项目根目录执行以下整段命令。Fold 1 完成后才启动 Fold 2；存在 checkpoint 时自动加 `--resume`，已经完成的折直接跳过。这里沿用当前实验命令的每批最多 10 次尝试；本节首次启动及后续续跑保持 `$maxAttempts` 相同。上面单折示例为 5 次，若已有实验是以 5 次启动，请将这里改为 5。

```powershell
$maxAttempts = 10
foreach ($fold in 1, 2) {
    $outputRoot = "docs/weibo21_adversarial_knowledge_fold_$fold/adversarial-evolve"
    $checkpoint = "$outputRoot/train-checkpoint.json"
    $hasCheckpoint = Test-Path -LiteralPath $checkpoint
    if ($hasCheckpoint) {
        $state = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $checkpoint)) | ConvertFrom-Json
        if ($state.status -eq 'completed') {
            Write-Host "Fold $fold already completed; skipping."
            continue
        }
    }
    $trainArgs = @(
        '--config', "configs/weibo21_adversarial_knowledge_fold_$fold.yaml",
        '--limit', '200', '--batch-size', '16', '--sample-concurrency', '8',
        '--progress', 'adversarial-evolve',
        '--max-attempts-per-batch', "$maxAttempts",
        '--max-supplement-rounds', '2',
        '--output', "$outputRoot/train.json"
    )
    if ($hasCheckpoint) { $trainArgs += '--resume' }
    Write-Host "Training fold $fold... resume=$hasCheckpoint"
    & .\evofact.cmd @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Fold $fold failed or was interrupted. Fix the cause, then rerun this block to resume."
    }
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Fold $fold did not save a checkpoint."
    }
    $state = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $checkpoint)) | ConvertFrom-Json
    if ($state.status -ne 'completed') {
        throw "Fold $fold is not complete: $($state.status). Fold 2 will not start."
    }
}
Write-Host 'Both folds have completed training.'
```

网络错误或手动中断后，修复原因并重新执行整段即可。此命令只训练，不自动启动 final-test；保持电脑通电且不睡眠。不要同时在其他终端启动同一折训练，以免共用技能库和断点。

## 自动依次测试两个 Fold（自动生成配置、自动续跑）

两折训练完成后，在项目根目录选择以下一条命令执行。脚本先检查两折训练断点均为 `completed` 且存在 active 技能库，再依次测试 Fold 1、Fold 2。25／50 条的测试配置自动生成，无需先执行后面的手动生成步骤。

每领域前 25 条（两折共 225 条）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/evaluate_adversarial_folds.ps1 -SamplesPerDomain 25
```

每领域前 50 条（两折共 450 条）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/evaluate_adversarial_folds.ps1 -SamplesPerDomain 50
```

原每领域 100 条（两折共 900 条）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/evaluate_adversarial_folds.ps1 -SamplesPerDomain 100
```

脚本使用 `--evolution-route adversarial-evolve`、batch=16、样本并发=4，读取每折当前已训练的 active 技能库。结果按规模分别写入 `final-test-25.*`、`final-test-50.*` 和原 `final-test.*`，不修改训练结果。

测试中断或存在失败样本时，重新执行同一条命令即可，脚本根据逐样本记录自动加 `--resume`，已完成样本不会重复调用模型；续跑期间保持模型、技能库和配置不变。单折测试失败会保留记录并继续另一折，最后列出未完成的折；手动中断则立即停止。若只测试 Fold 2，可在命令末尾添加 `-Folds 2`。不同规模分别重新推理，不共用逐样本记录；若仅需从完整 100 条结果统计前缀，可离线截取以避免重复调用。

## 每领域 25 条和 50 条 final-test

以下命令在训练完成后执行，使用原来的已训练技能库。每个领域固定读取原 `*-100.json` 的前 25 条或前 50 条，与从已有 100 条结果截取的样本一致。`--limit` 只控制训练数量，不能用来指定测试条数。

**执行顺序：先完成本路线训练，再执行下面的配置生成代码，最后选择 25 条或 50 条测试命令。** 只复制测试命令不会自动生成配置，会报 `FileNotFoundError`。测试前核对对应折 `train.json` 或 `train-checkpoint.json` 的 `status=completed`，以及本路线技能库中已存在 `active.json`；首次空技能库会从种子加载，不能当作自进化后的测试。

先生成独立测试配置，保留原配置及全部训练路径，只修改每领域测试数量。在项目根目录执行一次：

```powershell
$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($fold in 1, 2) {
    $source = "configs/weibo21_adversarial_knowledge_fold_$fold.yaml"
    $text = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $source), $utf8)
    $pattern = '(?m)^([ \t]*final_test_samples_per_domain:[ \t]*)100[ \t]*\r?$'
    if ([regex]::Matches($text, $pattern).Count -ne 1) { throw "Expected exactly one 100/domain setting: $source" }
    $targetRoot = "docs/weibo21_adversarial_knowledge_fold_$fold/adversarial-evolve"
    New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
    foreach ($n in 25, 50) {
        $config = [regex]::Replace($text, $pattern, ('${1}' + $n))
        [System.IO.File]::WriteAllText("$targetRoot/config.final-test-$n.yaml", $config, $utf8)
    }
}
```

测试命令须保留 `--evolution-route adversarial-evolve`，读取本路线已训练的技能库。

### final-test 25（每领域 25 条）

Fold 1 共 125 条，Fold 2 共 100 条，两折共 225 条：

```powershell
.\evofact.cmd --config docs/weibo21_adversarial_knowledge_fold_1/adversarial-evolve/config.final-test-25.yaml --evolution-route adversarial-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_adversarial_knowledge_fold_1/adversarial-evolve/final-test-25.json
.\evofact.cmd --config docs/weibo21_adversarial_knowledge_fold_2/adversarial-evolve/config.final-test-25.yaml --evolution-route adversarial-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_adversarial_knowledge_fold_2/adversarial-evolve/final-test-25.json
```

### final-test 50（每领域 50 条）

Fold 1 共 250 条，Fold 2 共 200 条，两折共 450 条：

```powershell
.\evofact.cmd --config docs/weibo21_adversarial_knowledge_fold_1/adversarial-evolve/config.final-test-50.yaml --evolution-route adversarial-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_adversarial_knowledge_fold_1/adversarial-evolve/final-test-50.json
.\evofact.cmd --config docs/weibo21_adversarial_knowledge_fold_2/adversarial-evolve/config.final-test-50.yaml --evolution-route adversarial-evolve --batch-size 16 --sample-concurrency 4 --progress test --output docs/weibo21_adversarial_knowledge_fold_2/adversarial-evolve/final-test-50.json
```

每条测试命令都会调用模型重新推理。若本路线已有完整的每领域 100 条结果且只需统计子集，可以从这份对抗实验结果中按原测试顺序截取前 25／50 条重新计算指标；普通 evolve 或无 evolve 已导出的结果仅用于比较，不能替代本路线的结果。25 条和 50 条分别生成 `final-test-25.*`、`final-test-50.*` 汇总、逐样本记录及测试断点，不覆盖原 `final-test.*` 或训练结果。测试中断后，在对应命令末尾添加 `--resume`，保持配置、模型和技能库不变；不要将不同测试规模的记录或断点混用。


## 续跑

中断后，在原训练命令末尾加 `--resume`。训练自动读取配置中的 checkpoint，复用已经保存的生成响应、核验结果和挑战 trace。保持配置、训练数量、batch、并发、验证领域、补充生成上限和输出路径不变。

以下续跑命令对应每批最多尝试 10 次的实验（与上面的自动两折命令一致）；若最初使用 5 次，请保持为 5。

Fold 1 续跑：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_knowledge_fold_1.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress adversarial-evolve --max-attempts-per-batch 10 --max-supplement-rounds 2 --output docs/weibo21_adversarial_knowledge_fold_1/adversarial-evolve/train.json --resume
```

Fold 2 续跑：

```powershell
.\evofact.cmd --config configs/weibo21_adversarial_knowledge_fold_2.yaml --limit 200 --batch-size 16 --sample-concurrency 8 --progress adversarial-evolve --max-attempts-per-batch 10 --max-supplement-rounds 2 --output docs/weibo21_adversarial_knowledge_fold_2/adversarial-evolve/train.json --resume
```

出现 `checkpoint already exists` 时应续跑，不要删除训练断点。`--resume` 不会修复网络连接；如果出现 `WinError 10061`，先排查 API 服务或代理连接，再续跑。进度条的 `16/16` 表示该批推理流程已结束，不能据此认定全部样本成功；错误详情以 `training-traces.jsonl` 和 checkpoint 为准。

当前 batch 达到尝试上限后，保留当前技能库，在 `skipped_batches` 记录 `reason=attempt_limit`，清空该批对抗生成缓存及临时验证反馈，自动进入下一批。旧版本停在 `status=attempt_limit` 的断点可直接使用原命令加 `--resume`，无需提高上限。`completed_batches` 包含无更新但已处理的批次。如果状态为 `validation_pool_exhausted`，相同配置的验证池已经不足，重复续跑不会产生新样本。训练达到 `completed` 后，再运行最终测试。

测试中断时，在原 `test` 命令末尾加 `--resume`。已完成样本跳过，失败或缺失样本补跑。

**测试命令必须保留 `--evolution-route adversarial-evolve`，才能读取对抗路线的技能库。** 普通 evolve 的两折测试脚本仅针对普通路线，不要用它代替这里的测试命令。

## 中间结果位置

两折分别写入：

- `docs/weibo21_adversarial_knowledge_fold_1/adversarial-evolve/`
- `docs/weibo21_adversarial_knowledge_fold_2/adversarial-evolve/`

每个目录中：

- `training-traces.jsonl`：真实训练、挑战推理及基线、候选验证 trace。
- `train-checkpoint.json`：批次、尝试、候选、生成响应、核验结果、挑战 trace、失败反馈及预算。
- `train.json`：训练命令正常返回时的汇总；中途异常以 checkpoint 为准。
- `adversarial-batch-report.json`：训练决策及生成轮次报告；`accepted_challenge_samples` 为核验通过的挑战总数。
- `generation-audit.json`：正常返回后输出的生成与核验审计；单轮接受数在 `rounds[].verification.accepted` 中。
- `final-test.json`：持续更新的独立测试汇总，未完成时指标只覆盖已完成样本。
- `final-test.samples.jsonl`：逐条刷盘的测试结果、trace 和错误记录，续跑时不能删除。
- `final-test.checkpoint.json`：测试进度与结果文件路径。

技能库分别位于 `skills/store_weibo21_adversarial_knowledge_fold_1/adversarial-evolve/` 和 `skills/store_weibo21_adversarial_knowledge_fold_2/adversarial-evolve/`。首次空库从 `skills/seeds/` 开始；已有库则继续使用 active bank。两折相互独立，也不读取普通 evolve 或 meta-evolve 的演化结果。

`generation.batch_size` 控制每轮挑战数；`execution.batch_size` 控制真实训练批次；`data.evolution_validation_samples` 控制每轮真实验证数。这三者不是同一个 batch，也不是对同一条样本重复推理。

训练样本 ID、领域配额、批次数和批次大小与普通 evolve 一致，两折均为 13 批；为满足跨域验证，具体 batch 组合及顺序不同。`training_sampling` 保存样本 ID、配额和 batch 划分。旧的单领域或固定留出领域 checkpoint 不能恢复到新布局；跑过旧方案时，请为新实验使用新的技能库和输出目录，不覆盖旧结果。
