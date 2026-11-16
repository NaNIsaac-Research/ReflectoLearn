# Runs G2c inside one test-queue job; finished draws and checkpointed fits are skipped, so a preempted run
# (exit 75) restarts where it stopped. Writes docs/fisher_global/g2c/run.done when every draw is written.
#   queue_guard.ps1 -Name xrr-g2c -JobCommand 'pwsh -NoProfile -File <this script>' -DoneFile docs/fisher_global/g2c/run.done
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:DEV 'NaNIsaac/_wt/ReflectoLearn-xrrpaper'
Set-Location $root
$exp = 'experiments/005_fisher_global'
$out = 'docs/fisher_global/g2c'
$log = Join-Path $out 'run_g2c'
New-Item -ItemType Directory -Force $log | Out-Null
& (Join-Path $exp 'run_workers.ps1') -Script "$exp/g2c_run.py" -Mode run -Workers 4 -LogDir $log -LockDir $out
# run_workers.ps1 sets an exit code only when preempted (75); in a fresh pwsh $LASTEXITCODE is otherwise $null
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$py = Join-Path $env:DEV 'NaNIsaac/_wt/ReflectoLearn-use-sim/.venv/Scripts/python.exe'
foreach ($k in 'tp', 'bp', 'tn') { "TEST-QUEUE-STAGE: G2c set $k, $([int](& $py -I "$exp/g2c_run.py" status --kind $k)) draws left" }
$left = [int](& $py -I "$exp/g2c_run.py" status)
if ($left -eq 0) { Set-Content (Join-Path $out 'run.done') (Get-Date -Format o) }
