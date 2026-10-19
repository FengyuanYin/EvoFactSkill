param(
    [ValidateSet(25, 50, 100)]
    [int]$SamplesPerDomain = 25,
    [ValidateSet(1, 2)]
    [int[]]$Folds = @(1, 2),
    [ValidateRange(1, 64)]
    [int]$SampleConcurrency = 4,
    [ValidateRange(0, 10)]
    [int]$SampleRetries = 2
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding($false)
$failedFolds = @()
$jobs = @()
Push-Location -LiteralPath $projectRoot
try {
    # Validate every selected fold before starting paid inference.
    foreach ($fold in $Folds) {
        $outputRoot = "docs/weibo21_adversarial_knowledge_fold_$fold/adversarial-evolve"
        $trainingCheckpoint = "$outputRoot/train-checkpoint.json"
        $activeBank = "skills/store_weibo21_adversarial_knowledge_fold_$fold/adversarial-evolve/active.json"
        if (-not (Test-Path -LiteralPath $trainingCheckpoint)) {
            throw "Fold ${fold}: training checkpoint not found: $trainingCheckpoint"
        }
        $state = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $trainingCheckpoint), $utf8) | ConvertFrom-Json
        if ($state.status -ne 'completed') {
            throw "Fold ${fold}: training is not completed (status=$($state.status))."
        }
        if (-not (Test-Path -LiteralPath $activeBank)) {
            throw "Fold ${fold}: trained active bank not found: $activeBank"
        }
        $source = "configs/weibo21_adversarial_knowledge_fold_$fold.yaml"
        $configPath = $source
        $stem = 'final-test'
        if ($SamplesPerDomain -ne 100) {
            $text = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $source), $utf8)
            $pattern = '(?m)^([ \t]*final_test_samples_per_domain:[ \t]*)100[ \t]*\r?$'
            if ([regex]::Matches($text, $pattern).Count -ne 1) {
                throw "Expected exactly one 100/domain setting: $source"
            }
            $config = [regex]::Replace($text, $pattern, ('${1}' + $SamplesPerDomain))
            $configPath = "$outputRoot/config.final-test-$SamplesPerDomain.yaml"
            if (Test-Path -LiteralPath $configPath) {
                $existing = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $configPath), $utf8)
                if ($existing -cne $config) {
                    throw "Existing test config differs; inspect it before resuming: $configPath"
                }
            }
            else {
                [System.IO.File]::WriteAllText((Join-Path $projectRoot $configPath), $config, $utf8)
            }
            $stem = "final-test-$SamplesPerDomain"
        }
        $jobs += [PSCustomObject]@{
            Fold = $fold
            Config = $configPath
            Output = "$outputRoot/$stem.json"
            Records = "$outputRoot/$stem.samples.jsonl"
            Checkpoint = "$outputRoot/$stem.checkpoint.json"
        }
    }
    foreach ($job in $jobs) {
        $evaluationArgs = @(
            '--config', $job.Config,
            '--evolution-route', 'adversarial-evolve',
            '--batch-size', '16', '--sample-concurrency', "$SampleConcurrency",
            '--progress', 'test', '--sample-retries', "$SampleRetries",
            '--output', $job.Output
        )
        $resume = Test-Path -LiteralPath $job.Records
        if ($resume) { $evaluationArgs += '--resume' }
        elseif (Test-Path -LiteralPath $job.Checkpoint) {
            throw "Test checkpoint exists without sample records. Restore the records: $($job.Records)"
        }
        Write-Host "Evaluating fold $($job.Fold), $SamplesPerDomain samples/domain... resume=$resume"
        & .\evofact.cmd @evaluationArgs
        if ($LASTEXITCODE -eq 130) {
            throw 'Evaluation interrupted. Rerun this script to resume.'
        }
        if ($LASTEXITCODE -ne 0) {
            $failedFolds += $job.Fold
            Write-Warning "Fold $($job.Fold) is incomplete. Results retained; continuing to the next fold."
            continue
        }
        $result = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $job.Output), $utf8) | ConvertFrom-Json
        if ($result.status -ne 'complete') {
            $failedFolds += $job.Fold
            Write-Warning "Fold $($job.Fold) returned status=$($result.status)."
        }
    }
}
finally {
    Pop-Location
}
if ($failedFolds.Count -gt 0) {
    throw "Incomplete folds: $($failedFolds -join ', '). Rerun the same command to resume."
}
Write-Host 'All selected folds have completed evaluation.'
