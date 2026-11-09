# Runs the stages of G1 deviation 1 in order inside one test-queue job; each stage skips finished results,
# so a preempted chain (exit 75) restarts where it stopped; a finished stage leaves wide/<mode>.done and is
# skipped on restart. Writes wide/chain.done at the end.
#   test-queue.ps1 -Name xrr-g1w -Preemptible -Command 'pwsh -NoProfile -File <this script>'
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:DEV 'NaNIsaac/_wt/ReflectoLearn-xrrpaper'
Set-Location $root
$exp = 'experiments/005_fisher_global'
$wide = 'docs/fisher_global/g1/wide'
$stages = @(
    @{ Script = "$exp/g1_wide.py"; Mode = 'xseed'; Workers = 3; Locks = "$wide/xseed"; Extra = '' },
    @{ Script = "$exp/g1_wide.py"; Mode = 'repeats'; Workers = 3; Locks = "$wide/repeats"; Extra = '' },
    @{ Script = "$exp/g1_wide.py"; Mode = 'm2'; Workers = 3; Locks = "$wide/m2"; Extra = '' },
    @{ Script = "$exp/g1_wide.py"; Mode = 'search'; Workers = 3; Locks = "$wide/search_M1W"; Extra = '' },
    @{ Script = "$exp/g1_hg1b.py"; Mode = 'profile'; Workers = 1; Locks = "$wide/hg1b_profiled"; Extra = '--variant wide' }
)
foreach ($s in $stages) {
    $done = Join-Path $wide "$($s.Mode).done"
    if (Test-Path $done) { continue }
    $log = Join-Path $wide "run_$($s.Mode)"
    New-Item -ItemType Directory -Force $log, $s.Locks | Out-Null
    & (Join-Path $exp 'run_workers.ps1') -Script $s.Script -Mode $s.Mode -Workers $s.Workers `
        -ExtraArgs $s.Extra -LogDir $log -LockDir $s.Locks
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Set-Content $done (Get-Date -Format o)
}
Set-Content (Join-Path $wide 'chain.done') (Get-Date -Format o)
