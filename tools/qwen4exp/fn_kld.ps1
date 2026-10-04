# [TAG_FN_SP0] Bounded Flash-Next KLD runs (E:/turbot-gates/flashnext/PLAN.md 7.6). GPU: only after SIX_DONE, ONE GPU process.
#
#   powershell -File tools/qwen4exp/fn_kld.ps1 -Arm base   -Corpus code  [-NcMoe 34] [-Chunks 2] [-Kv turbot]
#   powershell -File tools/qwen4exp/fn_kld.ps1 -Arm ncmoe40 -Corpus code -NcMoe 40            (placement-noise band)
#   powershell -File tools/qwen4exp/fn_kld.ps1 -Arm hot     -Corpus code -NcMoe 48 -Ub 3 -Chunks 2 -Env "LLAMA_MOE_HOT_PROFILE=E:\...\flashnext-q4.moeprof;LLAMA_MOE_HOT_MIB=14000"
#
# Caps (hard, not parameters): ctx 4096, at most 4 chunks. The base is written SPARSE (LLAMA_PPL_SPARSE_K=256, only the
# second half of each chunk scored): about 1 MB per chunk instead of the 2 GB per chunk of a dense 248K-vocab base (the
# dense base is what bugchecked the PC on 2026-09-22). Every non-base arm reads it and writes a per-token dump
# (LLAMA_PPL_KLD_DUMP) for the paired statistics (E:/turbot-gates/flashnext/turbot/fn_paired.py).
# Refuses: without SIX_DONE, with STOP_GPU or REBOOT_NEEDED, beside any llama/ggml/test process, with commit > 50 GB,
# with less than 30 GB free on E:. After the run: exit code, fatal messages, KLD summary, nvlddmkm/TDR events.
param(
  [Parameter(Mandatory=$true)][string]$Arm,
  [ValidateSet("code", "prose")][string]$Corpus = "code",
  [int]$NcMoe = 34,
  [int]$Chunks = 2,
  [int]$Ub = 512,
  [string]$Env = "",
  [string]$ExtraArgs = "",
  [string]$Model = "D:\Projects\LocalAI\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf",
  [string]$Bin = "D:\Projects\LocalAI\source-build\wt-flash\build-flash\bin",
  [string]$Kv = "turbot",  # [TAG_FN_MERGE] Flash-Next KV is turbot (owner 2026-09-28: never q4 / turbo4 KV)
  [string]$Out = "E:\turbot-gates\flashnext\kld"
)
$ErrorActionPreference = "Stop"
$Ctx = 4096
if ($Chunks -lt 1 -or $Chunks -gt 4) { "REFUSED: -Chunks must be 1..4 (cap 4 x 4096)"; exit 2 }
$Text = if ($Corpus -eq "code") { "E:\kv-bar-s0\code_corpus.txt" } else { "E:\kv-bar-s0\prose_corpus.txt" }
New-Item -ItemType Directory -Force $Out | Out-Null
# [TAG_FN_MERGE] the KV type is part of the base name: a candidate never reads a base of another KV type
$base = Join-Path $Out ("base_A_{0}_{1}_{2}x4096.sparse" -f $Kv, $Corpus, $Chunks)
$marker = "E:\turbot-gates\REBOOT_NEEDED.txt"

if (-not (Test-Path "E:\turbot-gates\imp\SIX_DONE")) { "REFUSED: E:\turbot-gates\imp\SIX_DONE missing"; exit 2 }
if (Test-Path "E:\turbot-gates\STOP_GPU") { "REFUSED: STOP_GPU exists"; exit 2 }
if (Test-Path $marker) { "REFUSED: $marker exists (GPU fault seen): reboot first"; exit 2 }
$busy = Get-Process | Where-Object { $_.ProcessName -match 'llama|ggml|test-backend|test-turbot|compute-sanitizer' }
if ($busy) { "REFUSED: running " + (($busy | ForEach-Object { "$($_.ProcessName) $($_.Id)" }) -join ", "); exit 2 }
$os = Get-CimInstance Win32_OperatingSystem
$commitGB = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB
if ($commitGB -gt 50) { "REFUSED: commit {0:N1} GB > 50 GB" -f $commitGB; exit 2 }
$freeE = (Get-PSDrive E).Free / 1GB
if ($freeE -lt 30) { "REFUSED: E: has {0:N1} GB free (< 30)" -f $freeE; exit 2 }
if ($Arm -ne "base" -and -not (Test-Path $base)) { "REFUSED: no base $base (run -Arm base first)"; exit 2 }
if ($Arm -eq "base" -and (Test-Path $base)) { "REFUSED: base $base exists; delete it by hand to rebuild"; exit 2 }

$log  = Join-Path $Out ("{0}_{1}_{2}_{3}x4096_ub{4}.log" -f $Arm, $Kv, $Corpus, $Chunks, $Ub)
$dump = Join-Path $Out ("{0}_{1}_{2}_{3}x4096_ub{4}.tok" -f $Arm, $Kv, $Corpus, $Chunks, $Ub)
$keep = @{}
$names = @("LLAMA_PPL_SPARSE_K", "LLAMA_PPL_SCORE_FIRST", "LLAMA_PPL_KLD_DUMP", "CUDA_VISIBLE_DEVICES")
$pairs = @()
if ($Env) { $pairs = $Env.Split(";") | Where-Object { $_ -match "=" } }
foreach ($p in $pairs) { $names += $p.Split("=", 2)[0] }
foreach ($k in $names) { $keep[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $null) }
foreach ($p in $pairs) { $kv = $p.Split("=", 2); [Environment]::SetEnvironmentVariable($kv[0], $kv[1]) }

$argv = @("-m", $Model, "-f", $Text, "-c", "$Ctx", "--chunks", "$Chunks", "-b", "4096", "-ub", "$Ub",
          "-ngl", "99", "--n-cpu-moe", "$NcMoe", "-fit", "off", "-fa", "on", "-ctk", $Kv, "-ctv", $Kv,
          "-t", "16", "--cpu-mask", "55555555", "--cpu-strict", "1")
if ($ExtraArgs) { $argv += $ExtraArgs.Split(" ") | Where-Object { $_ } }
if ($Arm -eq "base") {
  $env:LLAMA_PPL_SPARSE_K = "256"
  $argv += @("--kl-divergence-base", $base)
} else {
  $env:LLAMA_PPL_KLD_DUMP = $dump
  $argv += @("--kl-divergence-base", $base, "--kl-divergence")
}
$t0 = Get-Date
"[{0:HH:mm:ss}] $Arm $Corpus : llama-perplexity $($argv -join ' ')  env: $Env" -f $t0
$quoted = $argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
$p = Start-Process -FilePath "$Bin\llama-perplexity.exe" -ArgumentList $quoted -RedirectStandardOutput $log `
       -RedirectStandardError "$log.err" -NoNewWindow -Wait -PassThru
$rc = $p.ExitCode
foreach ($k in $keep.Keys) { [Environment]::SetEnvironmentVariable($k, $keep[$k]) }

$ev = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; StartTime=$t0.AddSeconds(-2)} -ErrorAction SilentlyContinue)
$ev += @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Display'; Id=4101; StartTime=$t0.AddSeconds(-2)} -ErrorAction SilentlyContinue)
if ($ev.Count -gt 0) {
  Add-Content $marker ("{0:s} REBOOT NEEDED: fn_kld {1}: events {2}" -f (Get-Date), $Arm, (($ev | ForEach-Object { $_.Id }) -join ","))
  "*** GPU EVENT during $Arm : reboot before anything else ***"; exit 3
}
$txt = (Get-Content $log -Raw) + "`n" + (Get-Content "$log.err" -Raw)
$crash = $txt -match "CUDA error|illegal memory access|GGML_ASSERT|ggml_abort"
$kld = [regex]::Match($txt, "Mean\s+KLD:\s*([0-9.eE+-]+)")
$top = [regex]::Match($txt, "Same top p:\s*([0-9.]+)")
$ppl = [regex]::Match($txt, "Mean\s+ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([0-9.eE+-]+)")
$ok = ($rc -eq 0) -and -not $crash
if ($Arm -ne "base") { $ok = $ok -and $kld.Success }
"{0,-14} {1,-5} rc {2} {3,5}s  KLD {4,-12} same-top {5,-8} ln-ppl-ratio {6,-12} {7}" -f $Arm, $Corpus, $rc,
  [int]((Get-Date) - $t0).TotalSeconds, $(if ($kld.Success) { $kld.Groups[1].Value } else { "-" }),
  $(if ($top.Success) { $top.Groups[1].Value } else { "-" }), $(if ($ppl.Success) { $ppl.Groups[1].Value } else { "-" }),
  $(if ($ok) { "OK" } else { "FAIL (see $log)" })
if (-not $ok) { exit 1 }
