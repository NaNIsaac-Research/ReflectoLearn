# Runs G2 inside one test-queue job; finished draws and checkpointed fits are skipped, so a preempted run
# (exit 75) restarts where it stopped. Writes docs/fisher_global/g2/run.done when every worker has finished.
#   queue_guard.ps1 -Name xrr-g2 -JobCommand 'pwsh -NoProfile -File <this script>' -DoneFile docs/fisher_global/g2/run.done
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:DEV 'NaNIsaac/_wt/ReflectoLearn-xrrpaper'
Set-Location $root
$exp = 'experiments/005_fisher_global'
$out = 'docs/fisher_global/g2'
$log = Join-Path $out 'run_g2'
New-Item -ItemType Directory -Force $log | Out-Null
& (Join-Path $exp 'run_workers.ps1') -Script "$exp/g2_run.py" -Mode run -Workers 4 -LogDir $log -LockDir $out
# run_workers.ps1 sets an exit code only when preempted (75); in a fresh pwsh $LASTEXITCODE is otherwise $null
if ($LASTEXITCODE) { exit $LASTEXITCODE }
# workers also stop at their wall-time deadline; only a run with every draw written is done
$py = Join-Path $env:DEV 'NaNIsaac/_wt/ReflectoLearn-use-sim/.venv/Scripts/python.exe'
$left = [int](& $py -I "$exp/g2_run.py" status)
if ($left -eq 0) { Set-Content (Join-Path $out 'run.done') (Get-Date -Format o) }
