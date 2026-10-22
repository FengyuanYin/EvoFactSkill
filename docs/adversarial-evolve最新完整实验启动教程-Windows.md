# 最新 adversarial-evolve 完整实验启动教程（Windows PowerShell）

适用于当前知识型对抗生成、REAL 训练锚点、多 Agent 联合优化、源领域验证完整反馈，以及 optimizer / generator 持久记忆与窗口管理版本。

**最新策略：新准备的 adversarial-evolve 配置默认 `independent_validation.source_feedback: true`。不再预留源领域独立保护名单，也不再运行第二阶段独立保护。所有符合跨域条件的源领域样本均可参与验证；每次候选完成固定三轮后，baseline / candidate trace、真实标签及判错诊断向 optimizer 开放，并跨批保存。Final-test 数据与标签继续隔离。**

**新实验的 optimizer 同时启用 `paired_regression_review_v1` 提示策略。** 目标选择和最终提案都要求配对审查“修正 / 退步 / 共同正确 / 共同错误”，将历史 REAL 误判与当前训练错误同等对待；增加规则前检查具有相同表面特征的正确反例，写清适用边界；重复已失败的假说必须给出新依据，否则返回 `no_change`。运行失败仍单独处理，现有 JSON 输出格式、联合修改权限、晋升规则和普通 evolve 提示词均不改变。

checkpoint 的 `optimizer_regression_policy` 保存版本和提示词摘要。旧 checkpoint 缺少该标记时继续使用原提示词，不自动升级；带标记的实验续跑必须使用相同提示版本。新的 `run.json` 也记录该策略，便于对照。下面新目录从当前初始种子开始，不复制旧实验进化后的技能库。

新建源领域完整反馈实验还默认启用 `paired_source_diagnostics_v1`：程序在同一次候选的所有验证轮结束后，按 `sample_id` 配对 baseline 和 candidate，生成 `paired_comparison`。其中列出真实标签、双方预测及 trace ID，统计修正、退步、共同正确、共同错误，并按真实标签和领域汇总；任一侧运行失败单独统计，不算修正或退步。净正确数变化为修正数减退步数，不代表 F1 增量。该诊断随验证反馈传给 optimizer，窗口裁剪完整 trace 时仍保留最近一次的配对表。晋升门控与验证样本选择保持原样；普通 evolve 不受影响。checkpoint 通过 `paired_source_diagnostics_policy` 固定此功能，缺少标记的旧断点续跑不自动添加诊断，也不改写历史结果。

旧目录没有这个配置项时，仍按原独立保护策略续跑。不要修改正在运行的配置，也不要将旧实验改成新策略后直接 `--resume`；配置身份不同会拒绝恢复。应用新策略请另开带时间标签的实验，保留旧结果。

**本教程所有训练和测试都写入新时间标签目录。旧实验的配置、checkpoint、技能库、trace 和 final-test 结果全部保留。不要删除旧文件，也不要将新命令指向 `docs/weibo21_adversarial_knowledge_fold_*` 或旧技能库。**

## 1. 本次设置

| 项目 | 设置 |
|---|---|
| 两折训练数据 | 沿用普通 evolve 的真实训练样本选择，每折 200 条 |
| 真实训练批次 | batch=16，共 13 批；按跨域验证要求组成混域批次 |
| 训练样本并发 | 8 |
| 源领域验证（新实验） | 固定 3 轮，每轮新增 8 条，共 24 条；来自当前批次未出现的源领域，按领域 / 标签分组交替取样 |
| 对抗生成（新实验） | 每轮最多 4 条；有判错时默认 3 条错误扩展、1 条边界探索，无判错则全部探索 |
| 补充触发（新实验） | 验证拒绝或相同信息连续两次 no_change；两者共享最多 5 轮补充，不计运行失败为攻击成功 |
| 生成材料 | 当前提供的带标签训练示例及模型内源知识；优先使用 REAL 锚点，保留 FAKE 参考 |
| 候选修改 | optimizer 自主选择一个或多个可编辑技能，可新增 specialist；整组验证与晋升 |
| 每批优化上限 | 总共最多尝试 5 次（含首次）；耗尽后记录跳过并进入下一批 |
| 验证候选范围 | 全部符合条件的源领域样本；不划出禁止反馈的保护池；直接训练仍固定 200 条 |
| 验证反馈 | 候选全部轮次结束后，开放双方 trace、真实标签与判错诊断；晋升和拒绝的经验都跨批保留 |
| 晋升条件 | 本次三轮聚合 Macro-F1 严格提升，整体及各域 FAKE 召回不下降，并通过安全检查；没有第二道独立保护否决 |
| 运行失败处理 | 仅重试运行失败样本，额外最多 2 次；完整输出但判错不重试 |
| optimizer / generator 记忆 | 均开启；保留跨批经验及单次请求的本地响应复用 |
| 上下文窗口 | 按本实验使用的 DeepSeek V4 Flash 设置，两者均为 1,000,000 tokens |
| 窗口预算 | 安全余量 10%，输出预留 8192 tokens；基础输入预算 891,808 tokens |

本地响应复用用于避免恢复时重复调用已经完成的请求，不等于保证服务端提示缓存命中。这里的“训练”更新技能文件，不更新模型权重。冻结的优化器自身、生成器、核验器、治理规则和标签契约不作为 optimizer 的修改目标。

8192 是初始输出额度，不是累计 token 预算。若服务端以 `finish_reason=length` 截断 JSON，记忆管理器会在剩余窗口、显式输出上限和已知服务端上限内提高额度；当前配置通常为 8192 → 16384 → 32768。输出扩容与输入溢出共用 `max_overflow_retries=2` 的请求内重试额度，重试用量均计入费用，断点续跑不重置已消耗的重试次数。若对抗生成仍被截断，该轮明确记录为生成失败、不接受半截样本；保留已有通过核验的挑战，继续使用训练反馈优化。该修复兼容本教程已有 checkpoint，无需改配置或删除结果。

### 固定多轮聚合验证

新实验同时启用错误驱动生成，checkpoint 记录 `adversarial_generation_policy`。生成前提供明确的判对/判错诊断，并固定每个生成位置的当前训练来源。生成器及盲核验器保持冻结，REAL / FAKE 标签不设配额。旧断点不升级；具体输入、补充触发及 `challenge_statistics` 统计口径见 [生成器机制与记忆说明](adversarial-generator记忆与上下文管理.md)。

新建 CLI 实验默认 `--validation-rounds 3`，同一候选在三轮内保持不变。验证前固定三轮的全部名单；每轮使用不同样本，baseline 和 candidate 使用相同名单。三轮结束后合并逐样本预测重新计算 Macro-F1、各域指标及 FAKE 召回，不平均各轮 F1，不根据中间结果提前晋升或临时追加样本。源领域验证是开发反馈，不作为独立泛化成绩；未见领域效果由 final-test 衡量。

拒绝后形成的新候选使用新的源领域验证样本，不沿用旧候选成绩。验证样本不足时明确停止，不能减少轮数或回退到当前训练批次的领域。可在首次启动时用 `--validation-rounds 1` 显式选择单轮对照；新实验续跑必须保持首次记录的轮数。完整验证反馈保存在 checkpoint 的 `source_validation_feedback` 中；每次请求按新到旧供 optimizer 使用，窗口不足时优先移除旧反馈，checkpoint 保留原始记录。

即使只剩最后一份历史验证反馈，其完整 trace 仍可逐条移出请求，优先移除判对案例，保留真实标签及判错诊断，再考虑减少当前 trace。没有配置 tokenizer 时使用 UTF-8 字节作保守估算，因此本地窗口不足不代表服务端实际 token 已达 1M。若旧断点停在 `protected optimizer input does not fit model window`，修复后直接使用原命令加 `--resume`；尚未调用模型的阶段会从原始请求重建上下文，已调用阶段的记录和预算不会重置，无需删除 checkpoint。

对抗生成的 API 读取超时会记录为该轮 `generation_error`，不接受不完整样本，也不计作攻击成功；保留之前通过核验的挑战，继续用现有训练反馈优化。失败调用仍计入预算，超时轮占用原有生成轮数，不额外无限重试。已因生成超时退出的旧断点可直接 `--resume`，已完成的训练和验证结果保留。

**本教程已有运行目录及旧 checkpoint 不自动升级。** 缺少 `validation_aggregation` 标记的断点仍按原单轮方式续跑，即使命令省略该参数。应用三轮策略需另开实验，保留旧结果；断点会保存全部轮次名单、逐样本进度和聚合结果，续跑只补未完成部分。

## 2. 使用已准备的新实验目录

在 PowerShell 中进入项目根目录：

```powershell
Set-Location 'F:\intern\Agent\auto-skill\EvoFactSkill'
$ErrorActionPreference = 'Stop'
$runRoot = 'docs/independent-validation/20261021-205918-b991fc2f'
```

上面的新目录已经离线准备好，两折均启用源领域验证完整反馈，尚未启动训练、没有调用模型，可直接进入第 3 步。`evofact.cmd` 使用项目内 `.venv` 的 Python，并沿用项目现有的 `.env`。推理模型别名为配置中的 `deepseek-flash`；optimizer 模型及端点使用 `EVOFACT_OPTIMIZER_MODEL`、`EVOFACT_OPTIMIZER_BASE_URL` 等现有环境设置。

本次文件位置：

```text
docs/independent-validation/20261021-205918-b991fc2f/
  run.json
  adversarial-evolve/
    fold_1/
      config.yaml
      validation-manifest.json
      adversarial-evolve/          # 本折训练、生成、测试输出
    fold_2/
      config.yaml
      validation-manifest.json
      adversarial-evolve/

skills/store_independent_validation/20261021-205918-b991fc2f/
  adversarial-evolve/
    fold_1/adversarial-evolve/     # 本折新技能库
    fold_2/adversarial-evolve/
```

两折首次运行从当前 `skills/seeds/` 初始化，各自进化，不复制以前训练得到的 active 技能。上述训练输出和技能目录由训练按需创建。

### 另开一次全新实验时使用

**仅另开实验时运行下面这段；使用上面的已准备目录时不需要运行，断点续跑也不能重新运行。** 准备脚本生成北京时间戳和随机短标签，不调用模型。对于 adversarial-evolve，默认不分配独立保护池，`--samples-per-domain 32` 与 `--evaluation-samples-per-domain 4` 只保留给旧保护模式及其他路线；新模式每轮仍由 `evolution_validation_samples: 8` 决定验证数量。需要保留旧独立保护作为新对照实验时，可额外添加 `--independent-protection`。

```powershell
$prepareArgs = @(
    'scripts/prepare_independent_validation_run.py',
    '--routes', 'adversarial-evolve',
    '--samples-per-domain', '32',
    '--evaluation-samples-per-domain', '4',
    '--optimizer-memory', '--optimizer-context-window', '1000000',
    '--generator-memory', '--generator-context-window', '1000000'
)
$prepareJson = & .\.venv\Scripts\python.exe @prepareArgs
if ($LASTEXITCODE -ne 0) { throw 'Experiment preparation failed; training has not started.' }
$prepared = ($prepareJson -join "`n") | ConvertFrom-Json
$runRoot = $prepared.run_root
Write-Host "Save this runRoot for resume: $runRoot"
```

以下所有命令都使用同一个 `$runRoot`。重新打开终端时，先进入项目根目录，再将它设回本次路径。

## 3. 自动依次训练 Fold 1、Fold 2（自动识别断点）

下面整段既是首次启动命令，也是中断后的续跑命令。Fold 1 完整处理后才启动 Fold 2；已完成的折会跳过。先核对两折配置的全部保存路径，才开始调用模型。

```powershell
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
if ($runRoot -notmatch '^docs/independent-validation/\d{8}-\d{6}-[0-9a-f]{8}$') {
    throw 'Set runRoot to the new isolated experiment directory from step 2.'
}
if (-not (Test-Path -LiteralPath "$runRoot/run.json")) { throw 'Run manifest not found.' }
$runTag = Split-Path -Leaf $runRoot
$maxAttempts = 5
$jobs = @()
foreach ($fold in 1, 2) {
    $foldRoot = "$runRoot/adversarial-evolve/fold_$fold"
    $outputRoot = "$foldRoot/adversarial-evolve"
    $configPath = "$foldRoot/config.yaml"
    $store = "skills/store_independent_validation/$runTag/adversarial-evolve/fold_$fold/adversarial-evolve"
    $text = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $configPath), $utf8)
    $expected = @{
        output_dir = $outputRoot
        skill_store = $store
        trace_log = "$outputRoot/training-traces.jsonl"
        checkpoint_path = "$outputRoot/train-checkpoint.json"
        store_path = "$outputRoot/generation-audit.json"
    }
    foreach ($key in $expected.Keys) {
        $pattern = '(?m)^[ \t]*' + [regex]::Escape($key) + ':[ \t]*' + [regex]::Escape($expected[$key]) + '[ \t]*\r?$'
        if ([regex]::Matches($text, $pattern).Count -ne 1) {
            throw "Fold ${fold}: unexpected $key; inspect the config before running."
        }
    }
    $jobs += [PSCustomObject]@{
        Fold = $fold; Config = $configPath; Output = $outputRoot
        Checkpoint = "$outputRoot/train-checkpoint.json"
    }
}
foreach ($job in $jobs) {
    $hasCheckpoint = Test-Path -LiteralPath $job.Checkpoint
    if ($hasCheckpoint) {
        $state = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $job.Checkpoint), $utf8) | ConvertFrom-Json
        if ($state.status -eq 'completed') {
            Write-Host "Fold $($job.Fold) already completed; skipping."
            continue
        }
    }
    $trainArgs = @(
        '--config', $job.Config,
        '--limit', '200', '--batch-size', '16', '--sample-concurrency', '8',
        '--progress', 'adversarial-evolve',
        '--max-attempts-per-batch', "$maxAttempts",
        '--max-supplement-rounds', '5',
        '--output', "$($job.Output)/train.json"
    )
    if ($hasCheckpoint) { $trainArgs += '--resume' }
    Write-Host "Training fold $($job.Fold)... resume=$hasCheckpoint"
    & .\evofact.cmd @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Fold $($job.Fold) stopped. Keep all files and rerun this block after resolving the cause."
    }
    $state = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $job.Checkpoint), $utf8) | ConvertFrom-Json
    if ($state.status -ne 'completed') {
        throw "Fold $($job.Fold) is not complete: $($state.status)."
    }
}
Write-Host 'Both folds have completed training.'
```

达到每批 5 次尝试上限会自动进入下一批，不会因为“没有晋升”停掉整个实验。`completed_batches=13` 表示 13 批都已处理，包含 `skipped_batches` 中没有晋升的批次，不表示技能更新了 13 次。

若出现 `optimizer_context_failure`、`generator_context_failure` 或运行失败状态，先检查原因；不清空目录，不重新准备实验。普通网络中断修复后可使用原配置续跑。若必须修改会影响实验身份的配置，另开新实验并保留旧目录。

### 单独首次训练一个 Fold

若只训练一折，完成第 2 步后执行下面这段即可，不执行上面的自动两折训练代码。`$fold = 1` 训练 Fold 1，改为 `2` 则只训练 Fold 2；其余参数与自动两折训练一致，输出仍在本次新目录中。

```powershell
$fold = 1
Set-Location 'F:\intern\Agent\auto-skill\EvoFactSkill'
$ErrorActionPreference = 'Stop'
$runRoot = 'docs/independent-validation/20261021-205918-b991fc2f'  # 另开实验时替换为实际新目录
$foldRoot = "$runRoot/adversarial-evolve/fold_$fold"
if (-not (Test-Path -LiteralPath "$foldRoot/config.yaml")) { throw "Config not found: $foldRoot/config.yaml" }
.\evofact.cmd --config "$foldRoot/config.yaml" --limit 200 --batch-size 16 --sample-concurrency 8 --progress adversarial-evolve --max-attempts-per-batch 5 --max-supplement-rounds 5 --output "$foldRoot/adversarial-evolve/train.json"
```

这是首次启动命令，不加 `--resume`；该折已有 checkpoint 时使用第 4 步的单折续跑命令，不删除原断点。

## 4. 中断与单折续跑

需要下班时使用 `Ctrl+C` 请求停止，等待进程退出。之后保留配置、技能库、checkpoint、生成缓存、逐样本记录和 trace；不要一边运行一边改配置或代码。恢复时使用相同版本的代码、模型设置和运行参数。

重新打开 PowerShell，先执行第 2 步的项目路径和 `$runRoot` 设置，再重跑第 3 步整段即可。也可以只恢复某一折：

```powershell
$fold = 1  # Fold 2 则改为 2
$foldRoot = "$runRoot/adversarial-evolve/fold_$fold"
.\evofact.cmd --config "$foldRoot/config.yaml" --limit 200 --batch-size 16 --sample-concurrency 8 --progress adversarial-evolve --max-attempts-per-batch 5 --max-supplement-rounds 5 --output "$foldRoot/adversarial-evolve/train.json" --resume
```

只有存在该折 `train-checkpoint.json` 时才加 `--resume`。对抗路线直接使用配置里的 `meta_learning.checkpoint_path`，此命令不需要额外的 `--checkpoint` 参数。

## 5. final-test：默认只测 Fold 1，25 / 50 / 100 分开保存

先完成训练。以下代码只读取本次新实验的 active 技能库。每个领域的 25 / 50 条固定取原 100 条静态名单的前缀，保持样本顺序；不是对全折总共只测试 25 / 50 条。

| 每领域样本数 | Fold 1 总量 | Fold 2 总量 | 输出 |
|---|---:|---:|---|
| 25 | 125 | 100 | `final-test-25.*` |
| 50 | 250 | 200 | `final-test-50.*` |
| 100 | 500 | 400 | `final-test.*` |

**本次默认只测试 Fold 1（`$testFolds = @(1)`），不会启动 Fold 2。需要两折时再改为 `@(1, 2)`。默认先跑 25。要跑 50，将 `$testSizes` 改成 `@(50)`；完整 100 改成 `@(100)`。也可设为 `@(25, 50, 100)`，依次分别推理，会产生三份独立调用费用。**

下面代码可单独复制执行，使用当前实验目录。选中的折必须已完成训练；若提示训练未完成，先续跑该折。不要在训练仍更新技能库时运行 final-test。

```powershell
Set-Location 'F:\intern\Agent\auto-skill\EvoFactSkill'
$runRoot = 'docs/independent-validation/20261021-205918-b991fc2f'
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$testFolds = @(1)
$testSizes = @(25)
if ($runRoot -notmatch '^docs/independent-validation/\d{8}-\d{6}-[0-9a-f]{8}$') { throw 'Invalid isolated runRoot.' }
if ($testFolds.Count -eq 0 -or @($testFolds | Where-Object { $_ -notin @(1, 2) }).Count -gt 0) { throw 'Choose fold 1 or 2.' }
if (@($testSizes | Where-Object { $_ -notin @(25, 50, 100) }).Count -gt 0) { throw 'Choose 25, 50 or 100.' }
$runTag = Split-Path -Leaf $runRoot
$jobs = @()
foreach ($n in $testSizes) {
    foreach ($fold in $testFolds) {
        $foldRoot = "$runRoot/adversarial-evolve/fold_$fold"
        $outputRoot = "$foldRoot/adversarial-evolve"
        $checkpoint = "$outputRoot/train-checkpoint.json"
        $state = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $checkpoint), $utf8) | ConvertFrom-Json
        if ($state.status -ne 'completed') { throw "Fold ${fold}: training is not completed." }
        $store = "skills/store_independent_validation/$runTag/adversarial-evolve/fold_$fold/adversarial-evolve"
        if (-not (Test-Path -LiteralPath "$store/active.json")) { throw "Fold ${fold}: trained active bank missing." }
        $source = "$foldRoot/config.yaml"
        $text = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $source), $utf8)
        foreach ($entry in @(@('output_dir', $outputRoot), @('skill_store', $store))) {
            $check = '(?m)^' + $entry[0] + ':[ \t]*' + [regex]::Escape($entry[1]) + '[ \t]*\r?$'
            if ([regex]::Matches($text, $check).Count -ne 1) { throw "Fold ${fold}: config points outside this run." }
        }
        $configPath = $source
        $stem = 'final-test'
        if ($n -ne 100) {
            $pattern = '(?m)^([ \t]*final_test_samples_per_domain:[ \t]*)100[ \t]*\r?$'
            if ([regex]::Matches($text, $pattern).Count -ne 1) { throw 'Expected one 100/domain setting.' }
            $testConfig = [regex]::Replace($text, $pattern, ('${1}' + $n))
            $configPath = "$foldRoot/config.final-test-$n.yaml"
            if (Test-Path -LiteralPath $configPath) {
                if ([IO.File]::ReadAllText((Resolve-Path -LiteralPath $configPath), $utf8) -cne $testConfig) {
                    throw "Existing test config differs; keep it and inspect: $configPath"
                }
            }
            else { [IO.File]::WriteAllText((Join-Path (Get-Location).Path $configPath), $testConfig, $utf8) }
            $stem = "final-test-$n"
        }
        $jobs += [PSCustomObject]@{
            Fold = $fold; Size = $n; Config = $configPath
            Output = "$outputRoot/$stem.json"
            Records = "$outputRoot/$stem.samples.jsonl"
            Checkpoint = "$outputRoot/$stem.checkpoint.json"
        }
    }
}
$failedJobs = @()
foreach ($job in $jobs) {
    $resume = Test-Path -LiteralPath $job.Records
    if (-not $resume -and ((Test-Path -LiteralPath $job.Checkpoint) -or (Test-Path -LiteralPath $job.Output))) {
        throw "Results exist without sample records; restore the records before resuming: $($job.Output)"
    }
    $testArgs = @(
        '--config', $job.Config, '--evolution-route', 'adversarial-evolve',
        '--batch-size', '16', '--sample-concurrency', '4',
        '--progress', 'test', '--sample-retries', '2', '--output', $job.Output
    )
    if ($resume) { $testArgs += '--resume' }
    Write-Host "Testing fold $($job.Fold), $($job.Size)/domain... resume=$resume"
    & .\evofact.cmd @testArgs
    if ($LASTEXITCODE -eq 130) { throw 'Test interrupted; rerun this block to resume.' }
    if ($LASTEXITCODE -ne 0) {
        $failedJobs += "fold=$($job.Fold),size=$($job.Size)"
        continue
    }
    $result = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $job.Output), $utf8) | ConvertFrom-Json
    if ($result.status -ne 'complete') { $failedJobs += "fold=$($job.Fold),size=$($job.Size)" }
}
if ($failedJobs.Count -gt 0) { throw "Incomplete tests retained: $($failedJobs -join '; '). Rerun to resume." }
Write-Host 'All requested tests completed.'
```

测试中断后保持 `$testSizes`、配置、模型和技能库不变，重跑整段自动续跑。不要删除 `.samples.jsonl` 或 `.checkpoint.json`。本教程不用旧的 `scripts/evaluate_adversarial_folds.ps1`，因为该脚本绑定的是旧实验路径。

## 6. 如何确认结果

每折的 `$runRoot/adversarial-evolve/fold_<n>/adversarial-evolve/` 中：

| 文件 | 用途 |
|---|---|
| `train-checkpoint.json` | 实时状态、批次、尝试、技能、记忆、预算及续跑信息；训练完成时 `status=completed` |
| `train.json` | 命令返回后的训练汇总 |
| `training-traces.jsonl` | 真实训练、挑战推理及源领域验证的 trace；旧保护模式还记录独立验证 |
| `adversarial-batch-report.json` | 优化尝试、门控结果和生成轮次 |
| `generation-audit.json` | 生成响应、核验结果和接受 / 拒绝记录 |
| `final-test-25.json` / `final-test-50.json` / `final-test.json` | 各规模测试汇总，完成时 `status=complete` |
| 对应 `.samples.jsonl` / `.checkpoint.json` | 测试逐样本结果与恢复进度 |

训练 checkpoint 的 `joint_optimization=true` 表示使用新联合提案模式。完整修改组查看 `attempts[].proposals` 和候选 `packages`，不要只看兼容旧格式的单数 `proposal`。模型生成了候选不代表已增强：只有核验通过并完成挑战推理的样本才能进入优化反馈。

新模式的 `validation-manifest.json` 记录全部源领域验证候选 ID，不保留独立保护池。评估完成后，双方 trace、真实标签和判错诊断存入 checkpoint 的 `source_validation_feedback`，跨批供 optimizer 使用。Final-test 始终不进入这些反馈。只有显式选择旧保护模式时，独立样本的文本、标签和逐样本预测才继续隔离。

不要将旧实验的训练结果或 final-test 文件复制到本次输出目录冒充新结果。旧结果继续保留在各自原目录，可在本次完成后用于比较。
