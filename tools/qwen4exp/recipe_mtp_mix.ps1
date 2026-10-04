# [TAG_FN_R4_MTP_MIX] Recipe "MTPmix": file A with the MTP head's 512 routed experts requantized from q8_0 to the trunk's own
# mix (gate/up q4_K, down q5_1; mtp_requant.py picks the trunk's most common mix), the dense nextn block layout of file A
# (merge_mtp.py --mtp-ratio 0). Frees ~1.03 GiB of VRAM for the hot set; only the drafts can change (the target verifies
# every token with its unchanged weights). The A/B against file A is the acceptance and the speed at 32K / 262K.
#
#   powershell -File tools\qwen4exp\recipe_mtp_mix.ps1 [-Force] [-DryRun] [-Bin <dir with ggml-base.dll>]
#
# CPU and disk only: reads the 2.6 GB head, writes a ~1.6 GB head and a 5-shard set whose shards 2-4 are hardlinks of the
# downloaded trunk (no copy) and whose shard 5 is the new head. It refuses while a llama / test / build / sanitizer process
# runs or commit charge is high (it must not overlap a GPU speed measurement), and never touches the source files.
param([switch]$Force, [switch]$DryRun, [string]$Bin = "D:\Projects\LocalAI\source-build\wt-fsync\build-fsync\bin")
$ErrorActionPreference = "Stop"

$M       = "D:\Projects\LocalAI\models"
$Tools   = $PSScriptRoot
$Head    = "$M\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
$HeadMix = "$M\mtp-Qwen3.8-Flash-Next-shared-trunkmix.gguf"
$SrcPat  = "$M\Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf"
$A1      = "$M\Qwen3.8-Flash-Next-UD-Q4_K_XL-MTP-00001-of-00005.gguf"
$MixPat  = "$M\Qwen3.8-Flash-Next-UD-Q4_K_XL-MTPmix-%05d-of-00005.gguf"
$Out     = "E:\turbot-gates\flashnext\test\r4\recipe_mtp_mix"
New-Item -ItemType Directory -Force $Out | Out-Null

# inputs present
foreach ($f in @($Head, $A1, $SrcPat.Replace("%05d", "00001"))) {
  if (-not (Test-Path $f)) { throw "missing input $f" }
}

$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "-1"

# 1. the plan (header reads only), and the tensor-type file of the same plan for the llama-quantize route
& python "$Tools\mtp_requant.py" --head $Head --trunk $A1 --dry-run --tt-out "$Out\tt_mtp48_trunkmix.txt" *>&1 | Tee-Object "$Out\plan.log"
if ($LASTEXITCODE -ne 0) { throw "mtp_requant.py --dry-run failed ($LASTEXITCODE)" }
if ($DryRun) { "dry run only: $Out\plan.log"; exit 0 }

# guards before any write: a quiet machine, commit charge, disk space, the quantizer
$busy = Get-Process | Where-Object { $_.ProcessName -match 'llama|ggml|test-|compute-sanitizer|cmake|ninja|nvcc|^cl$|cicc|^link$' }
if ($busy) { throw ("REFUSED: busy: " + (($busy | ForEach-Object { "$($_.ProcessName) $($_.Id)" }) -join ", ")) }
$os = Get-CimInstance Win32_OperatingSystem
$commitGB = ($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB
if ($commitGB -gt 50) { throw ("REFUSED: commit {0:N1} GB > 50" -f $commitGB) }
$freeD = (Get-PSDrive D).Free / 1GB
if ($freeD -lt 10) { throw ("REFUSED: D: has {0:N0} GB free, 10 GB needed" -f $freeD) }
if (-not (Test-Path "$Bin\ggml-base.dll")) { throw "missing $Bin\ggml-base.dll (-Bin)" }

# 2. the requantized head (ggml's quantizer, verified: tensor set, types, byte-equal rest, requantization error)
$argv = @("$Tools\mtp_requant.py", "--head", $Head, "--trunk", $A1, "--out", $HeadMix, "--ggml", $Bin, "--threads", "8")
if ($Force) { $argv += "--force" }
& python @argv *>&1 | Tee-Object "$Out\requant.log"
if ($LASTEXITCODE -ne 0) { throw "mtp_requant.py failed ($LASTEXITCODE), see $Out\requant.log" }

# 3. merged like file A: dense nextn block (ratio 0), shards 2-4 hardlinked, shard 5 = the new head (sha1-verified)
$argv = @("$Tools\merge_mtp.py", "--src-pattern", $SrcPat, "--n-src", "4", "--head", $HeadMix, "--out-pattern", $MixPat,
          "--mtp-ratio", "0")
if ($Force) { $argv += "--force" }
& python @argv *>&1 | Tee-Object "$Out\merge.log"
if ($LASTEXITCODE -ne 0) { throw "merge_mtp.py failed ($LASTEXITCODE), see $Out\merge.log" }

# 4. the type report: trunk as file A, blk.48 gate/up q4_K and down q5_1 (expect C)
& python "$Tools\fn_types.py" $MixPat.Replace("%05d", "00001") --expect C --json "$Out\types_mix.json" *>&1 | Tee-Object "$Out\types.log"
if ($LASTEXITCODE -ne 0) { throw "type check failed, see $Out\types.log" }
"MTPmix ready: " + $MixPat.Replace("%05d", "00001")
