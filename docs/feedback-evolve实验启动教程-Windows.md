# feedback-evolve 两折实验命令（Windows PowerShell）

对应 [普通 evolve 两折教程](普通evolve实验启动教程-Windows.md)，不是 valid=16 教程。在项目根目录执行，命令自动读取本地 `.env`。

沿用原教程的两折配置、领域划分、seed、模型、预算和门控。训练使用 `--limit 200 --batch-size 16 --sample-concurrency 8`，测试使用 `--batch-size 16 --sample-concurrency 4`。验证数量从原配置继承，当前两折均为 8，不改成 16。

新增差异是 feedback-evolve 的批内验证 trace 反馈和重试，默认每批最多 5 次。每次候选验证使用新的样本，训练与验证样本保持分离；同批训练 trace 复用。因此候选数、验证样本消耗、时间和成本会随反馈机制变化。

## 先生成本次配置和保存目录

每次开启新实验执行一次，续跑不执行。北京时间戳加随机短标签用于区分每次启动。现有两折 YAML 只读；新配置仅重写四个保存路径，其他配置原样保留。

```powershell
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$workspace = (Get-Location).Path
$chinaZone = [TimeZoneInfo]::FindSystemTimeZoneById('China Standard Time')
$startedAt = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $chinaZone)
$runTag = $startedAt.ToString('yyyyMMdd-HHmmss-fff') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runRoot = "docs/feedback-evolve/$runTag"
$storeRoot = "skills/store_feedback_evolve/$runTag"
if ((Test-Path -LiteralPath $runRoot) -or (Test-Path -LiteralPath $storeRoot)) {
    throw 'Run directory already exists; generate a new run tag.'
}
$templates = @{}
$sourceDigests = [ordered]@{}
foreach ($fold in @(1, 2)) {
    $source = Join-Path $workspace "configs/weibo21_evolve_fold_$fold.yaml"
    $text = [IO.File]::ReadAllText($source, $utf8)
    foreach ($pattern in @('(?m)^output_dir:.*$', '(?m)^skill_store:.*$', '(?m)^  trace_log:.*$', '(?m)^  checkpoint_path:.*$')) {
        if ([regex]::Matches($text, $pattern).Count -ne 1) {
            throw "Unexpected source config field: $pattern in $source"
        }
    }
    $templates[$fold] = $text
    $sourceDigests["fold_$fold"] = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
}
foreach ($fold in @(1, 2)) {
    $foldDir = "$runRoot/fold_$fold"
    $outputDir = "$foldDir/feedback-evolve"
    $skillDir = "$storeRoot/fold_$fold/feedback-evolve"
    New-Item -ItemType Directory -Path $outputDir -ErrorAction Stop | Out-Null
    New-Item -ItemType Directory -Path $skillDir -ErrorAction Stop | Out-Null
    $text = $templates[$fold]
    $text = [regex]::Replace($text, '(?m)^output_dir:.*$', "output_dir: $outputDir")
    $text = [regex]::Replace($text, '(?m)^skill_store:.*$', "skill_store: $skillDir")
    $text = [regex]::Replace($text, '(?m)^  trace_log:.*$', "  trace_log: $outputDir/training-traces.jsonl")
    $text = [regex]::Replace($text, '(?m)^  checkpoint_path:.*$', "  checkpoint_path: $outputDir/meta-checkpoint.json")
    [IO.File]::WriteAllText((Join-Path $workspace "$foldDir/config.yaml"), $text, $utf8)
}
$manifest = [ordered]@{
    mode = 'feedback-evolve'
    run_tag = $runTag
    started_at_beijing = $startedAt.ToString('yyyy-MM-ddTHH:mm:ss') + '+08:00'
    output_root = $runRoot
    skill_store_root = $storeRoot
    source_config_sha256 = $sourceDigests
    training_limit = 200
    training_batch_size = 16
    training_sample_concurrency = 8
    test_batch_size = 16
    test_sample_concurrency = 4
    max_attempts_per_batch = 5
}
[IO.File]::WriteAllText((Join-Path $workspace "$runRoot/run.json"), ($manifest | ConvertTo-Json -Depth 4), $utf8)
Write-Host "Run tag: $runTag"
Write-Host "Output: $runRoot"
Write-Host "Stores: $storeRoot"
```

新技能库从当前 `skills/seeds/` 开始，两折互不共享，也不复制普通实验训练后的 active store。原配置和普通实验的训练结果均保留。上面的代码只生成配置，不启动训练。

## Fold 1

在生成配置的同一个窗口中训练：

```powershell
.\evofact.cmd --config "$runRoot/fold_1/config.yaml" --limit 200 --batch-size 16 --sample-concurrency 8 --progress feedback-evolve --max-attempts-per-batch 5 --checkpoint "$runRoot/fold_1/feedback-evolve/feedback-evolve-checkpoint.json" --output "$runRoot/fold_1/feedback-evolve/feedback-evolve.json"
```

训练完成后测试：

```powershell
.\evofact.cmd --config "$runRoot/fold_1/config.yaml" --evolution-route feedback-evolve --batch-size 16 --sample-concurrency 4 --progress test --output "$runRoot/fold_1/feedback-evolve/final-test.json"
```

## Fold 2

训练：

```powershell
.\evofact.cmd --config "$runRoot/fold_2/config.yaml" --limit 200 --batch-size 16 --sample-concurrency 8 --progress feedback-evolve --max-attempts-per-batch 5 --checkpoint "$runRoot/fold_2/feedback-evolve/feedback-evolve-checkpoint.json" --output "$runRoot/fold_2/feedback-evolve/feedback-evolve.json"
```

训练完成后测试：

```powershell
.\evofact.cmd --config "$runRoot/fold_2/config.yaml" --evolution-route feedback-evolve --batch-size 16 --sample-concurrency 4 --progress test --output "$runRoot/fold_2/feedback-evolve/final-test.json"
```

最终测试沿用 `outputs/static-100-before-pr-review/` 中原配置对应领域的 `*-100.json`，保持样本 ID 和顺序。Fold 1 测试 500 条，Fold 2 测试 400 条。测试须指定 `--evolution-route feedback-evolve`，读取本路线技能库；原 `scripts/evaluate_evolve_folds.ps1` 仍用于普通 evolve，不用于本次实验。

## 续跑同一次实验

如果窗口关闭，先恢复本次标签：

```powershell
$runTag = '<feedback-run-tag>'
$runRoot = "docs/feedback-evolve/$runTag"
if (-not (Test-Path -LiteralPath "$runRoot/run.json")) { throw 'Run manifest not found.' }
```

对需要恢复的折，重新执行上面的对应训练或测试命令，并在末尾加 `--resume`。不要重新生成目录、修改配置、删除逐样本记录或复制其他实验的技能库。

- `status=completed`：该折所有训练批已完成，可以测试。
- `attempt_limit`：当前批达到累计尝试上限，保存后停止，返回非零退出码。若需要继续，将 `--max-attempts-per-batch 5` 改为更高的累计上限（如 10），并加 `--resume`；其余参数保持不变。
- `validation_pool_exhausted`：不足一组新的验证样本，保存后停止；提高尝试上限不能解决，也不会循环复用已验证样本。

新断点格式为 `feedback_evolve_checkpoint_v1`。普通实验的 `evolve_checkpoint_v1` 继续通过原 `evolve --resume` 使用，两者不混用。测试的网络错误重试和失败样本续跑沿用原测试机制。

## 中间结果位置

| 内容 | 位置 |
|---|---|
| 时间标签、原配置摘要和启动参数 | `docs/feedback-evolve/<tag>/run.json` |
| 两折完整新配置 | `docs/feedback-evolve/<tag>/fold_<n>/config.yaml` |
| 训练 trace、断点、尝试记录和验证曲线 | `docs/feedback-evolve/<tag>/fold_<n>/feedback-evolve/` |
| `final-test.json`、`final-test.samples.jsonl`、`final-test.checkpoint.json` | 同上目录 |
| 两折独立技能库 | `skills/store_feedback_evolve/<tag>/fold_<n>/feedback-evolve/` |

本教程对齐你指定的普通教程及当前两折配置。磁盘上旧的已完成 Fold 1 训练记录曾使用 valid=50、并发 4，Fold 2 使用 valid=8、并发 8；历史结果与当前教程参数不能直接视为完全相同设置。比较实验时以各次 checkpoint 中实际记录为准。
