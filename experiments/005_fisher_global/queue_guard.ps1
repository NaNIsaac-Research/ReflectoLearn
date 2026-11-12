param(
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$JobCommand,
    [Parameter(Mandatory)][string]$DoneFile,
    [string]$Log = 'queue_guard.log',
    [string]$NotifyTo = 'xrr'
)
# Keeps a long G-run polite: when another run (pr-land, a lane's full suite) is waiting behind this test-queue job,
# the job is stopped (finished results stay, workers skip them on resume), and it is started again through
# test-queue once the queue is free. The job is queued -Preemptible, so a pr-land preempts it by itself (run_workers.ps1
# exits 75); this guard is the fallback and only acts on a waiter that has stayed 5 minutes. Ends when DoneFile exists.
$queue = Join-Path $env:DEV 'NaNIsaac/DotFiles/herdr/tools/test-queue.ps1'
$me = $PID
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $Log }
function Mine {
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $me -and $_.CommandLine -and (
            $_.CommandLine -match "test-queue\.ps1.*-Name $Name" -or
            $_.CommandLine -match 'run_workers\.ps1' -or
            $_.CommandLine -match 'g1_run\.py|g2c_run\.py|g2_run\.py|g3_run\.py')
    }
}
function Waiters {
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $me -and $_.Name -eq 'pwsh.exe' -and $_.CommandLine -match 'test-queue\.ps1' -and
        $_.CommandLine -notmatch "-Name $Name" -and $_.CommandLine -notmatch '-Status' -and
        $_.CommandLine -notmatch 'queue_guard'
    }
}
function Start-JobQueued {
    Start-Process pwsh -ArgumentList '-NoProfile', '-File', $queue, '-Name', $Name, '-Preemptible', '-Command', "`"$JobCommand`"" -WindowStyle Hidden
    Log 'job started'
}
Start-Sleep 20
while (-not (Test-Path $DoneFile)) {
    $running = @(Mine)
    if (@(Waiters).Count -eq 0) { $firstSeen = $null } elseif (-not $firstSeen) { $firstSeen = Get-Date }
    if ($running.Count -gt 0 -and $firstSeen -and ((Get-Date) - $firstSeen).TotalMinutes -ge 5) {
        $firstSeen = $null
        $w = (Waiters | Select-Object -First 1).CommandLine
        Log "waiter seen: $($w.Substring(0, [math]::Min(120, $w.Length))); pausing"
        $running | Where-Object Name -match 'python' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        $running | Where-Object Name -match 'pwsh' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep 20
        do { Start-Sleep 30 } until ((& pwsh -NoProfile -File $queue -Status) -eq 'free' -and @(Waiters).Count -eq 0)
        Start-Sleep 60
        if ((& pwsh -NoProfile -File $queue -Status) -eq 'free') { Start-JobQueued }
    } elseif ($running.Count -eq 0) {
        if ((& pwsh -NoProfile -File $queue -Status) -eq 'free' -and @(Waiters).Count -eq 0) { Start-JobQueued; Start-Sleep 60 }
    }
    Start-Sleep 15
}
Log 'done file present; guard ends'
# the lane that started the run has no watch of its own; tell it the job is done
& courier send $NotifyTo "[queue_guard] $Name done: $DoneFile" 2>&1 | Out-Null
if ($LASTEXITCODE) { Log "courier send $NotifyTo failed (exit $LASTEXITCODE)" }
