# [TAG_FN_MOE_TRACE] Day-1 routing trace for Flash-Next (E:/turbot-gates/flashnext/PLAN.md F3 "Trace"), then the offline replay.
# GPU: after SIX_DONE only, ONE GPU process at a time (refuses beside any llama/ggml/test process).
#
#   powershell -File tools/moe-trace/collect_traces.ps1 [-NcMoe 38] [-Tokens 20480] [-Agentic <file>]
#
# 1. teacher forcing: llama-perplexity over ~20K tokens each of code, prose and (if given) agentic / tool-call text, with
#    LLAMA_MOE_TRACE (+ LLAMA_MOE_TRACE_PRED=1 for the next-layer recall) -> <Out>/<domain>.moet
# 2. real decode with MTP: fn_bench.py-style server run, ~2K generated tokens (code greedy + prose temp 1), with the trace
#    -> <Out>/decode.moet (verify-window statistics, U(k) at the real widths)
# 3. offline, CPU only: make_profile.py -> flashnext-q4.moeprof (all domains, prefill windows of 3 as decode steps, plus
#    the real decode), route_sim.py -> summary.json (static / LRU / windowed-LFU / Belady hit rates at the 131K and 262K
#    budgets, alpha, U(k), recall) - decision D3
param(
  [int]$NcMoe = 38,
  [int]$Tokens = 20480,
  [string]$Agentic = "",
  [string]$Model = "D:\Projects\LocalAI\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf",
  [string]$Bin = "D:\Projects\LocalAI\source-build\wt-fsync\build-fsync\bin",  # [TAG_SYNC_1004] flashnext/synced
  [string]$Out = "E:\turbot-gates\flashnext\route",
  [string]$Budgets = "14300,16000"
)
$ErrorActionPreference = "Stop"
$Tree = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path   # [TAG_SYNC_1004] the tools of this checkout
if (-not (Test-Path "E:\turbot-gates\imp\SIX_DONE")) { "REFUSED: E:\turbot-gates\imp\SIX_DONE missing"; exit 2 }
if (Test-Path "E:\turbot-gates\STOP_GPU") { "REFUSED: STOP_GPU exists"; exit 2 }
New-Item -ItemType Directory -Force $Out | Out-Null

function Assert-Quiet {
  $busy = Get-Process | Where-Object { $_.ProcessName -match 'llama|ggml|test-backend|test-turbot|compute-sanitizer' }
  if ($busy) { throw ("REFUSED: running " + (($busy | ForEach-Object { "$($_.ProcessName) $($_.Id)" }) -join ", ")) }
}

$corpora = [ordered]@{ code = "E:\kv-bar-s0\code_corpus.txt"; prose = "E:\kv-bar-s0\prose_corpus.txt" }
if ($Agentic) { $corpora["agentic"] = $Agentic }
$chunks = [math]::Max(1, [int]($Tokens / 4096))

foreach ($dom in $corpora.Keys) {
  Assert-Quiet
  $trace = Join-Path $Out "$dom.moet"
  $log   = Join-Path $Out "trace_$dom.log"
  $keep = @{}
  foreach ($k in @("LLAMA_MOE_TRACE", "LLAMA_MOE_TRACE_PRED", "LLAMA_MOE_PROFILE", "LLAMA_MOE_TRACE_SOURCE")) {
    $keep[$k] = [Environment]::GetEnvironmentVariable($k)
  }
  $env:LLAMA_MOE_TRACE = $trace
  $env:LLAMA_MOE_TRACE_PRED = "1"
  $env:LLAMA_MOE_PROFILE = Join-Path $Out "$dom.moeprof"
  $env:LLAMA_MOE_TRACE_SOURCE = "teacher-forced $dom"
  $argv = @("-m", $Model, "-f", $corpora[$dom], "-c", "4096", "--chunks", "$chunks", "-b", "4096", "-ub", "512",
            "-ngl", "99", "--n-cpu-moe", "$NcMoe", "-fit", "off", "-fa", "on", "-ctk", "turbo4", "-ctv", "turbo4",
            "-t", "16", "--cpu-mask", "55555555", "--cpu-strict", "1")
  "[{0:HH:mm:ss}] trace $dom : $chunks x 4096 tokens" -f (Get-Date)
  $p = Start-Process -FilePath "$Bin\llama-perplexity.exe" -ArgumentList $argv -RedirectStandardOutput $log `
         -RedirectStandardError "$log.err" -NoNewWindow -Wait -PassThru
  foreach ($k in $keep.Keys) { [Environment]::SetEnvironmentVariable($k, $keep[$k]) }
  if ($p.ExitCode -ne 0 -or -not (Test-Path $trace)) { "trace $dom FAILED (rc $($p.ExitCode)), see $log"; exit 1 }
  "  {0}: {1:N1} MB" -f $trace, ((Get-Item $trace).Length / 1MB)
}

# real decode with MTP through the server: one fn_bench arm with the trace switched on
Assert-Quiet
$arms = Join-Path $Out "arms_trace.json"
$decTrace = (Join-Path $Out "decode.moet").Replace("\", "/")
@"
[{"name": "trace", "env": {"LLAMA_MOE_TRACE": "$decTrace", "LLAMA_MOE_TRACE_PRED": "1", "LLAMA_MOE_TRACE_SOURCE": "decode-mtp"}}]
"@ | Set-Content -Encoding ascii $arms
& python "$Tree\tools\qwen4exp\fn_bench.py" --arms $arms --tag trace_decode --reps 1 --n 1024 --warmup 64 --out $Out `
    --base-args ("-c 131072 -ngl 99 --n-cpu-moe $NcMoe -fit off -fa on -ctk turbo4 -ctv turbo4 -b 2048 -ub 512 -t 16 " +
                 "--cpu-mask 55555555 --cpu-strict 1 --prio 2 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.5")
if ($LASTEXITCODE -ne 0) { "decode trace FAILED"; exit 1 }

# offline replay (CPU only)
$types = "E:\turbot-gates\flashnext\recipe\types_A.json"
if (-not (Test-Path $types)) {
  & python "$Tree\tools\qwen4exp\fn_types.py" $Model --json $types | Out-Null
}
$traceArgs = @()
foreach ($dom in $corpora.Keys) { $traceArgs += @("--trace", (Join-Path $Out "$dom.moet")) }
$traceArgs += @("--trace", (Join-Path $Out "decode.moet"))
& python "$Tree\tools\moe-trace\make_profile.py" @traceArgs --prefill-as-decode 3 --out (Join-Path $Out "flashnext-q4.moeprof") `
    --source "code+prose+agentic+decode" --plan-mib ($Budgets.Split(",")[0]) --types $types --host-layers 0-47
$simArgs = @()
foreach ($dom in $corpora.Keys) { $simArgs += @("--trace", ("{0}={1}" -f $dom, (Join-Path $Out "$dom.moet"))) }
$simArgs += @("--trace", ("decode={0}" -f (Join-Path $Out "decode.moet")))
& python "$Tree\tools\moe-trace\route_sim.py" @simArgs --types $types --budget-mib $Budgets --out $Out
"done: $Out\flashnext-q4.moeprof and $Out\summary.json"
