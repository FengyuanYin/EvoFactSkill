# 独立均衡验证与 FAKE 召回保护（Windows PowerShell）

如需同时启用跨批 optimizer 记忆和 token 窗口管理，使用 [Optimizer 持久记忆教程](optimizer持久记忆与上下文管理.md) 中的新配置目录，并将本文循环的 `$runRoot` 替换为该目录。本文原目录继续保留原行为。

本方案用于新的 feedback-evolve、meta-evolve、adversarial-evolve 实验。原配置默认关闭此功能，普通 evolve 的流程保持原样。旧实验断点、技能库与结果保留；新策略改变数据划分与晋升标准，必须新建实验，不能接续旧断点。

## 验证规则

- 每个源领域固定预留 32 条独立样本，REAL、FAKE 各 16 条，固定 seed=42。训练的 200 个样本 ID 保持原方案，独立池排除训练样本、最终测试样本以及共享事件、来源谱系和近重复正文。
- 独立池不进入训练、Generator 输入或 Optimizer 的 trace/反馈。运行断点中保留完整审计，优化器只看到拒绝原因，不接收独立样本的正文、标签或预测明细。
- meta-evolve、adversarial-evolve 仍使用混域训练 batch，独立验证只选本批未出现的源领域。候选池保持每域真假各 16 条，每次实际从每个入选领域抽 4 条，REAL、FAKE 各 2 条；没有跨域子集时直接报错，绝不回退到本批领域或最终测试领域。
- feedback-evolve 保持普通样本划分方式：从原 evolution/protected 源分区中预留独立池，从原批内验证池中排除独立样本及近重复。其余 protected 样本不会变成批内优化反馈。
- 原批内验证仍为每次 8 条、取新样本、不放回。候选先满足原来的批内 Macro-F1 提升，再进入独立验证。独立池固定复用，不能与批内“取新样本”混为一谈。

## 晋升保护

候选必须同时满足：

1. 批内 Macro-F1 严格提高，安全检查通过。
2. 本次抽取的独立验证子集上整体 Macro-F1 不下降。
3. 独立验证上整体及每个入选领域的 FAKE 召回率不下降（阈值 `max_positive_recall_drop: 0.0`）。实际实现使用标签契约定义的正类，Weibo21 对应 FAKE。
4. 独立验证中任一入选领域 Macro-F1 降幅不超过 5 个百分点（`max_domain_f1_drop: 0.05`）。

训练、挑战推理、批内验证及独立验证发生超时、网络/HTTP 错误、JSON/标签解析错误、运行时弃权等情况时，先单独重试失败样本，默认最多再试 2 次。成功完成但判错的样本正常计分，不为追求正确答案重复采样。重试及失败单独审计；重试后仍失败则保存断点并返回非零退出码，不对这次候选作指标晋升判断。续跑只补当前阶段未成功的样本，保留已成功样本。

独立验证仅在批内候选通过后运行；feedback-evolve 每轮从全部源领域各取 4 条，meta/adversarial 仅从本批未训练领域各取 4 条。例如本批留出两个源领域时，每侧验证 8 条，baseline 与 candidate 合计 16 次样本推理。

`samples_per_domain: 32` 控制固定候选池，`evaluation_samples_per_domain: 4` 控制每次实际验证量。候选池的 ID、训练 200 条及原批内 8 条验证均保持不变。每次抽样由固定 seed 与候选验证位置确定，真假均衡；同一候选的两侧使用相同 ID，并写入 checkpoint 供续跑复用，不要求不同候选之间不放回。原配置未设置新字段时仍验证完整跨域子集，旧断点行为保持不变。

## 已准备的实验

本次已生成：

```powershell
$runRoot = 'docs/independent-validation/20261019-102854-0e39d943'
```

下面的准备命令仅在需要创建另一组全新实验时执行，输出新的时间标签，不调用模型、不启动训练：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_independent_validation_run.py --samples-per-domain 32 --evaluation-samples-per-domain 4
```

准备脚本先核验所有领域配额，再生成三条路线各两折的 `config.yaml` 与 `validation-manifest.json`。每份名单记录 200 个训练 ID、独立池的 ID/领域/标签，以及每批允许使用的跨域验证候选 ID（不是每次实际抽取的 4 条名单）。两折分别预留 128 和 160 条独立样本。meta-evolve 与 adversarial-evolve 的训练、独立池名单一致，便于对照。本次新配置保留原候选池，只添加每次抽取 4 条的设置；原 `20261019-094036-c5cee853` 配置与结果保留，不能混接其旧训练断点。

## 自动训练所选路线的两折

在项目根目录执行整段，`$route` 可改为 `meta-evolve` 或 `feedback-evolve`：

```powershell
$runRoot = 'docs/independent-validation/20261019-102854-0e39d943'
$route = 'adversarial-evolve'
$maxAttempts = 5
foreach ($fold in 1, 2) {
    $foldRoot = "$runRoot/$route/fold_$fold"
    $outputRoot = "$foldRoot/$route"
    $checkpoint = "$outputRoot/train-checkpoint.json"
    $hasCheckpoint = Test-Path -LiteralPath $checkpoint
    if ($hasCheckpoint) {
        $state = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $checkpoint)) | ConvertFrom-Json
        if ($state.status -eq 'completed') { Write-Host "Fold $fold completed; skipping."; continue }
    }
    $trainArgs = @(
        '--config', "$foldRoot/config.yaml", '--limit', '200',
        '--batch-size', '16', '--sample-concurrency', '8', '--progress', $route,
        '--max-attempts-per-batch', "$maxAttempts", '--output', "$outputRoot/train.json"
    )
    if ($route -eq 'adversarial-evolve') { $trainArgs += @('--max-supplement-rounds', '2') }
    if ($route -eq 'feedback-evolve') { $trainArgs += @('--checkpoint', $checkpoint) }
    if ($hasCheckpoint) { $trainArgs += '--resume' }
    & .\evofact.cmd @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Fold $fold stopped. Fix runtime errors, then rerun this block to resume." }
    $state = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $checkpoint)) | ConvertFrom-Json
    if ($state.status -ne 'completed') { throw "Fold $fold is incomplete: $($state.status)" }
}
```

`training_runtime_failure`、`challenge_runtime_failure`、`validation_runtime_failure` 均表示基础设施/推理执行失败尚未恢复，不表示分类能力低或候选被拒绝。修复原因后执行同一段命令自动续跑。不要同时在两个终端训练同一折。

## 连续测试每领域 25 条

两折训练完成后，在同一窗口执行；切换路线时同步修改 `$route`。25、50、100 条分别测试，不混用记录。

```powershell
$samplesPerDomain = 25
if ($samplesPerDomain -notin 25, 50, 100) { throw 'Choose 25, 50 or 100.' }
$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($fold in 1, 2) {
    $foldRoot = "$runRoot/$route/fold_$fold"
    $outputRoot = "$foldRoot/$route"
    $state = [IO.File]::ReadAllText((Resolve-Path "$outputRoot/train-checkpoint.json")) | ConvertFrom-Json
    if ($state.status -ne 'completed') { throw "Fold $fold training not completed." }
    $text = [IO.File]::ReadAllText((Resolve-Path "$foldRoot/config.yaml"), $utf8)
    $pattern = '(?m)^([ \t]*final_test_samples_per_domain:[ \t]*)100[ \t]*\r?$'
    if ([regex]::Matches($text, $pattern).Count -ne 1) { throw 'Expected one 100/domain setting.' }
    $config = [regex]::Replace($text, $pattern, ('${1}' + $samplesPerDomain))
    $testConfig = "$foldRoot/config.final-test-$samplesPerDomain.yaml"
    [IO.File]::WriteAllText($testConfig, $config, $utf8)
    $testArgs = @('--config', $testConfig, '--evolution-route', $route,
        '--batch-size', '16', '--sample-concurrency', '4', '--progress', 'test',
        '--output', "$outputRoot/final-test-$samplesPerDomain.json")
    if (Test-Path "$outputRoot/final-test-$samplesPerDomain.samples.jsonl") { $testArgs += '--resume' }
    & .\evofact.cmd @testArgs
    if ($LASTEXITCODE -ne 0) { throw "Fold $fold test incomplete; rerun to resume." }
}
```

测试仍固定读取原每领域 100 条名单的前 N 条，使用对应新实验的 active 技能库。独立验证只用于训练晋升，不改变最终测试样本或测试指标口径。
