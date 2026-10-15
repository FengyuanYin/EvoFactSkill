param(
    [ValidateSet(1, 2)]
    [int[]]$Folds = @(1, 2),
    [ValidateRange(1, 64)]
    [int]$SampleConcurrency = 4,
    [ValidateRange(0, 10)]
    [int]$SampleRetries = 2
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
$failedFolds = @()
try {
    foreach ($fold in $Folds) {
        $output = "docs/weibo21_evolve_fold_$fold/final-test.json"
        $records = "docs/weibo21_evolve_fold_$fold/final-test.samples.jsonl"
        $evaluationArgs = @(
            '--config', "configs/weibo21_evolve_fold_$fold.yaml",
            '--batch-size', '16',
            '--sample-concurrency', "$SampleConcurrency",
            '--progress', 'test',
            '--sample-retries', "$SampleRetries",
            '--output', $output
        )
        if (Test-Path -LiteralPath $records) {
            $evaluationArgs += '--resume'
        }
        Write-Host "Evaluating fold $fold..."
        & .\evofact.cmd @evaluationArgs
        if ($LASTEXITCODE -eq 130) {
            throw 'Evaluation interrupted. Saved samples are retained; rerun this script to resume.'
        }
        if ($LASTEXITCODE -ne 0) {
            $failedFolds += $fold
            Write-Warning "Fold $fold is incomplete. Saved samples are retained; continuing to the next fold."
        }
    }
}
finally {
    Pop-Location
}
if ($failedFolds.Count -gt 0) {
    throw "Incomplete folds: $($failedFolds -join ', '). Run this script again to resume."
}
Write-Host 'Both selected folds have completed evaluation.'
