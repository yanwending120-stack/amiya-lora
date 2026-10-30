# 从 ASTR 镜像批量下载明日方舟剧情原文 JSON
# 用法: pwsh -File 01_download_story.ps1 [-Parallel 8]
param(
    [int]$Parallel = 8,
    [string]$Root = './raw'
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

Write-Host "[1/3] 取文件清单……"
$h = @{ 'User-Agent' = 'curl' }
$tree = Invoke-RestMethod -Headers $h `
    -Uri 'https://api.github.com/repos/050644zf/ArknightsStoryJson/git/trees/main?recursive=1' `
    -TimeoutSec 90
$paths = $tree.tree |
    Where-Object { $_.path -like 'zh_CN/gamedata/story/*.json' } |
    Select-Object -ExpandProperty path
Write-Host "      共 $($paths.Count) 个剧情文件"

New-Item -ItemType Directory -Force -Path $Root | Out-Null

Write-Host "[2/3] 下载（并发 $Parallel）……"
$sw = [Diagnostics.Stopwatch]::StartNew()
$done = 0
$results = $paths | ForEach-Object -Parallel {
    $p = $_
    $rel = $p -replace '^zh_CN/gamedata/story/', ''
    $out = Join-Path $using:Root $rel
    $dir = Split-Path $out -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if ((Test-Path $out) -and (Get-Item $out).Length -gt 100) { return @{ ok = $true; path = $p; skip = $true } }
    $url = 'https://r2.m31ns.top/' + $p
    curl.exe -s --fail --retry 2 --connect-timeout 10 -m 120 -o $out $url 2>$null
    $ok = ($LASTEXITCODE -eq 0) -and (Test-Path $out) -and ((Get-Item $out).Length -gt 100)
    return @{ ok = $ok; path = $p; skip = $false }
} -ThrottleLimit $Parallel

$fail = $results | Where-Object { -not $_.ok }
$skip = ($results | Where-Object { $_.skip }).Count
$sw.Stop()
Write-Host ("      完成 {0}，跳过（已存在）{1}，失败 {2}，耗时 {3:N0} 秒" -f `
    ($results.Count - $fail.Count), $skip, $fail.Count, $sw.Elapsed.TotalSeconds)
if ($fail.Count -gt 0) {
    $fail | Select-Object -ExpandProperty path | Set-Content -Path (Join-Path $Root '..\failed.txt') -Encoding utf8
    Write-Host "      失败清单已写入 failed.txt"
}

Write-Host "[3/3] 统计下载量……"
$files = Get-ChildItem $Root -Recurse -File -Filter *.json
$sum = ($files | Measure-Object Length -Sum).Sum
Write-Host ("      本地 {0} 个文件，共 {1:N1} MB" -f $files.Count, ($sum / 1MB))
