param(
    [Parameter(Mandatory)][string]$Script,
    [Parameter(Mandatory)][string]$Mode,
    [int]$Workers = 2,
    [string]$ExtraArgs = '',
    [string]$LogDir = '.'
)
# Runs `Workers` BelowNormal worker processes of a G-run script inside one test-queue job and logs the peak
# working set of each. Meant to be called as: test-queue.ps1 -Name xrr-g2b -Command '<this script and args>'.
$py = Join-Path $env:DEV 'NaNIsaac/_wt/ReflectoLearn-use-sim/.venv/Scripts/python.exe'
$env:XRR_IN_QUEUE = '1'
$t0 = Get-Date
$procs = @()
foreach ($w in 0..($Workers - 1)) {
    $argList = @('-I', $Script, $Mode, '--worker', "$w", '--workers', "$Workers") + ($ExtraArgs -split ' ' | Where-Object { $_ })
    $p = Start-Process $py -ArgumentList $argList -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $LogDir "worker$w.log") -RedirectStandardError (Join-Path $LogDir "worker$w.err")
    $p.PriorityClass = 'BelowNormal'
    $procs += $p
}
$peak = @(0) * $Workers
$cpu = @(0.0) * $Workers
while ($procs | Where-Object { -not $_.HasExited }) {
    for ($i = 0; $i -lt $Workers; $i++) {
        $p = $procs[$i]
        if (-not $p.HasExited) {
            # the venv launcher spawns the real interpreter as a child; measure the whole tree
            $kids = Get-Process -Id (Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)").ProcessId -ErrorAction SilentlyContinue
            foreach ($k in @($p) + @($kids)) {
                $k.Refresh()
                if ($k.PeakWorkingSet64 -gt $peak[$i]) { $peak[$i] = $k.PeakWorkingSet64 }
                if ($k.TotalProcessorTime.TotalHours -gt $cpu[$i]) { $cpu[$i] = $k.TotalProcessorTime.TotalHours }
            }
        }
    }
    Start-Sleep -Seconds 10
}
$wall = ((Get-Date) - $t0).TotalHours
$summary = [ordered]@{
    wall_hours = [math]::Round($wall, 3)
    cpu_hours = @($cpu | ForEach-Object { [math]::Round($_, 3) })
    peak_rss_gb = @($peak | ForEach-Object { [math]::Round($_ / 1GB, 2) })
}
$summary | ConvertTo-Json | Set-Content (Join-Path $LogDir 'resources.json') -Encoding utf8
$summary | ConvertTo-Json
