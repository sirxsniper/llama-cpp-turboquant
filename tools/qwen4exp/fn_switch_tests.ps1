# [TAG_FN_SP0] Unit-level checks of the Flash-Next speed switches (flashnext/speed), no real model. GPU: after SIX_DONE only.
#
#   powershell -File tools/qwen4exp/fn_switch_tests.ps1 [-Bin D:\Projects\LocalAI\source-build\wt-fsync\build-fsync\bin]
#
# 1. test-backend-ops -b CUDA0 on the new op shapes: the hot chain (remap + zero slot, T 1..8), the QSA-chunk top-k /
#    concat / cast shapes, and the MTP head-rows shapes
# 2. test-llama-archs -a qwen4exp (trunk) and --mtp (MTP head) once with every switch off and once per switch, so each
#    switched graph still runs and matches the CPU backend with random weights (the switches are read once per process);
#    skipped on Windows shared builds, where tests/CMakeLists.txt does not build test-llama-archs
# 3. python tools/moe-trace/route_sim.py --selftest
# Refuses beside any llama/ggml/test process; stops at the first failure.
param(
  [string]$Bin = "D:\Projects\LocalAI\source-build\wt-fsync\build-fsync\bin",  # [TAG_SYNC_1004] the synced branch (flashnext/synced)
  [string]$Out = "E:\turbot-gates\flashnext\switch_tests"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path "E:\turbot-gates\imp\SIX_DONE")) { "REFUSED: E:\turbot-gates\imp\SIX_DONE missing"; exit 2 }
if (Test-Path "E:\turbot-gates\STOP_GPU") { "REFUSED: STOP_GPU exists"; exit 2 }
$busy = Get-Process | Where-Object { $_.ProcessName -match 'llama|ggml|test-backend|test-turbot|compute-sanitizer' }
if ($busy) { "REFUSED: running " + (($busy | ForEach-Object { "$($_.ProcessName) $($_.Id)" }) -join ", "); exit 2 }
New-Item -ItemType Directory -Force $Out | Out-Null

function Run([string]$name, [string]$exe, [string[]]$argv, [hashtable]$envs) {
  $keep = @{}
  foreach ($k in $envs.Keys) { $keep[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $envs[$k]) }
  $log = Join-Path $Out "$name.log"
  $quoted = $argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
  $p = Start-Process -FilePath $exe -ArgumentList $quoted -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -Wait -PassThru
  foreach ($k in $keep.Keys) { [Environment]::SetEnvironmentVariable($k, $keep[$k]) }
  $txt = (Get-Content $log -Raw) + (Get-Content "$log.err" -Raw)
  $bad = ($p.ExitCode -ne 0) -or ($txt -match "CUDA error|illegal memory access|GGML_ASSERT|FAIL")
  "{0,-34} rc {1}  {2}" -f $name, $p.ExitCode, $(if ($bad) { "FAIL (see $log)" } else { "OK" })
  if ($bad) { exit 1 }
}

$tbo = Join-Path $Bin "test-backend-ops.exe"
# [TAG_FN_R4_QSA_POS] [TAG_FN_R4_BRIDGE_DMA] round 4: the QSA mask from positions (kv 4096 .. 262144) and the bridge's
# fetch (PCIe share: copy, slot ids, bank chain, timeout and refused-plan paths). Their first GPU run belongs under
# compute-sanitizer memcheck (fn_switch_tests runs them plain).
Run "tbo_qsa_mask"    $tbo @("-b", "CUDA0", "-o", "QSA_MASK") @{}
Run "tbo_moe_fetch"   $tbo @("-b", "CUDA0", "-o", "MOE_HOST_FETCH") @{}
Run "tbo_hot_chain"   $tbo @("-b", "CUDA0", "-o", "MUL_MAT_ID", "-p", "n_hot=") @{}
Run "tbo_topk"        $tbo @("-b", "CUDA0", "-o", "TOP_K", "-p", "k=2051") @{}
Run "tbo_concat"      $tbo @("-b", "CUDA0", "-o", "CONCAT") @{}
Run "tbo_set_rows"    $tbo @("-b", "CUDA0", "-o", "SET_ROWS", "-p", "248320") @{}
Run "tbo_mul_mat"     $tbo @("-b", "CUDA0", "-o", "MUL_MAT", "-p", "m=98304") @{}
# [TAG_MOE_DMA_SHARE] [TAG_FN_PREFILL_STREAM] the gen5 plan/fence/gate logic on the CPU backend (no GPU); the GPU side is
# tools/qwen4exp/fn_gen5_check.py
$g5 = Join-Path $Bin "test-moe-gen5.exe"
if (Test-Path $g5) { Run "moe_gen5_cpu" $g5 @("--cpu") @{ CUDA_VISIBLE_DEVICES = "-1" } }
# [TAG_FN_R4_QSA_POS] [TAG_FN_R4_ADAPT_DECAY] [TAG_FN_R4_VNNI] CPU-only round-4 tests
foreach ($t in @("test-qsa-pos", "test-moe-decay")) {
  $exe = Join-Path $Bin "$t.exe"
  if (Test-Path $exe) { Run $t $exe @() @{ CUDA_VISIBLE_DEVICES = "-1" } }
}
$tqf = Join-Path $Bin "test-quantize-fns.exe"
if (Test-Path $tqf) { Run "quantize_fns_vnni" $tqf @() @{ CUDA_VISIBLE_DEVICES = "-1"; GGML_CPU_VNNI = "1" } }

$tla = Join-Path $Bin "test-llama-archs.exe"
$switches = [ordered]@{
  "off"         = @{}
  "split_async" = @{ GGML_SCHED_SPLIT_ASYNC = "1" }
  "ple_host"    = @{ LLAMA_PLE_HOST_GATHER = "1" }
  "per_width"   = @{ LLAMA_GRAPH_PER_WIDTH = "1" }
  "qsa_chunk"   = @{ TURBO_QSA_CHUNK = "2" }
  "head_rows"   = @{ LLAMA_MTP_HEAD_ROWS = "64" }
  "qsa_pos"     = @{ LLAMA_QSA_POS_MASK = "1" }                              # [TAG_FN_R4_QSA_POS]
  "qsa_pos_ch"  = @{ LLAMA_QSA_POS_MASK = "1"; LLAMA_QSA_POS_CHUNK = "16" }  # [TAG_FN_R4_QSA_POS]
  "mtp_window"  = @{ LLAMA_MTP_WINDOW = "16" }                               # [TAG_FN_R4_MTP_WINDOW]
  "trace"       = @{ LLAMA_MOE_PROFILE = (Join-Path $Out "archs.moeprof"); LLAMA_MOE_TRACE = (Join-Path $Out "archs.moet"); LLAMA_MOE_TRACE_PRED = "1" }
  "all"         = @{ GGML_SCHED_SPLIT_ASYNC = "1"; LLAMA_PLE_HOST_GATHER = "1"; LLAMA_GRAPH_PER_WIDTH = "1"; TURBO_QSA_CHUNK = "2"; LLAMA_MTP_HEAD_ROWS = "64";
                    LLAMA_QSA_POS_MASK = "1"; LLAMA_QSA_POS_CHUNK = "16"; GGML_CUDA_GRAPH_POKE = "1" }
}
if (Test-Path $tla) {
  foreach ($s in $switches.Keys) {
    Run "archs_$s"     $tla @("-a", "qwen4exp") $switches[$s]
    Run "archs_mtp_$s" $tla @("-a", "qwen4exp", "--mtp") $switches[$s]
  }
} else {
  # tests/CMakeLists.txt builds test-llama-archs only when NOT (WIN32 and BUILD_SHARED_LIBS); use fn_synth_check.py
  "test-llama-archs.exe not built (Windows shared build): run tools\qwen4exp\fn_synth_check.py for the qwen4exp graphs"
}
# [TAG_FN_MTP_COST] the MTP draft-length policy (CPU only). test-llama-archs --mtp above also checks LLAMA_MTP_HEAD_IDS and
# LLAMA_MTP_ATTN_WINDOW by itself whenever no MTP head switch is set ([TAG_FN_MTP_HEAD_IDS] [TAG_FN_MTP_ATTN_WINDOW])
$tmc = Join-Path $Bin "test-mtp-cost.exe"
if (Test-Path $tmc) { Run "mtp_cost" $tmc @() @{} }
# [TAG_FN_MERGE] the tools of this checkout, not of the round-1 worktree
$mt = Join-Path $PSScriptRoot "..\moe-trace"
& python (Join-Path $mt "route_sim.py") --selftest
if ($LASTEXITCODE -ne 0) { "route_sim selftest FAILED"; exit 1 }
if (Test-Path (Join-Path $Out "archs.moet")) {
  & python (Join-Path $mt "make_profile.py") --trace (Join-Path $Out "archs.moet") --out (Join-Path $Out "archs_from_trace.moeprof")
}
"ALL SWITCH TESTS PASSED"
