#!/bin/bash
# SMEM Pre-Dequant Benchmark — M5 Max
# Tests SMEM vs baseline at multiple context depths
#
# BEFORE RUNNING:
#   1. cd to the root of this repository
#   2. git checkout experiment/smem-pre-dequant
#   3. Build WITHOUT SMEM first (baseline):
#      cmake --build build -j12
#   4. Run: MODEL=<model.gguf> ./scripts/bench-smem-m5.sh baseline
#   5. Build WITH SMEM:
#      TURBO_SMEM_DEQUANT=1 cmake --build build -j12
#   6. Run: MODEL=<model.gguf> ./scripts/bench-smem-m5.sh smem
#
# Uses Qwen3.5-35B-A3B Q8_0 as MODEL (MoE, fits in memory, attention-heavy)

set -e

LABEL="${1:-baseline}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT/build/bin/llama-bench}"
MODEL="${MODEL:?set MODEL to the GGUF to benchmark, e.g. Qwen3.5-35B-A3B-Q8_0.gguf}"
OUTFILE="$ROOT/bench-smem-m5-${LABEL}.txt"

# KV depth before each pp512 / tg128 test (llama-bench -d)
DEPTHS=(0 8192 16384 32768)
KV_TYPES=("turbo3" "turbo4" "q8_0")

echo "=== SMEM M5 Benchmark: ${LABEL} ===" | tee "$OUTFILE"
echo "Model: $(basename "$MODEL")" | tee -a "$OUTFILE"
echo "Date: $(date)" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

for ctk in "${KV_TYPES[@]}"; do
    for d in "${DEPTHS[@]}"; do
        if [[ "$ctk" == "q8_0" && "$LABEL" == "smem" ]]; then
            echo "SKIP: q8_0 + smem (q8_0 unaffected by SMEM)" | tee -a "$OUTFILE"
            continue
        fi

        echo "--- ${ctk} @ depth ${d} ---" | tee -a "$OUTFILE"

        ctv="$ctk"
        "$LLAMA_BENCH" \
            -m "$MODEL" \
            -ngl 99 -fa 1 \
            -ctk "$ctk" -ctv "$ctv" \
            -t 1 \
            -p 512 -n 128 -d "$d" \
            2>&1 | tee -a "$OUTFILE"

        echo "" | tee -a "$OUTFILE"
    done
done

echo "=== Done: ${LABEL} ===" | tee -a "$OUTFILE"
echo "Results saved to: $OUTFILE"
