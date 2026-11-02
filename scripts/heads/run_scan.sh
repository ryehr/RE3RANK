#!/bin/bash
# Head scan for one backbone: traces (vLLM) and attention masses (HF), one shard per GPU, then the
# gold-mass ranking and the zero-training R@3 ranking of single heads and head pairs.
#   BASE=Qwen/Qwen3-4B bash scripts/heads/run_scan.sh
# Seed sample: 400 items each from MuSiQue and NarrativeQA (SETS / PER_SET to change).
set -u
cd "$(dirname "$0")/../.." || exit 9
BASE=${BASE:?backbone, e.g. Qwen/Qwen3-4B}
SETS=${SETS:-musique,narrativeqa}; PER_SET=${PER_SET:-400}
OUT=${OUT:-${RE3RANK_OUT:-outputs}/heads/$(basename "$BASE")}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}; GPUS=$(seq -s " " 0 $((NGPU-1)))
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"
if ! ls "$OUT"/traces.s*.json >/dev/null 2>&1; then
  for s in $GPUS; do
    CUDA_VISIBLE_DEVICES=$s python scripts/heads/scan_heads.py gen --model "$BASE" --out-dir "$OUT" \
      --sets "$SETS" --per-set "$PER_SET" --shard $s --nshards $NGPU > "$OUT/logs/gen.s$s.log" 2>&1 &
  done
  wait
  for s in $GPUS; do grep -q kept "$OUT/logs/gen.s$s.log" || { echo "trace generation failed: $OUT/logs/gen.s$s.log"; exit 1; }; done
fi
for s in $GPUS; do
  CUDA_VISIBLE_DEVICES=$s python scripts/heads/scan_heads.py scan --model "$BASE" --out-dir "$OUT" \
    --shard $s --nshards $NGPU > "$OUT/logs/scan.s$s.log" 2>&1 &
done
wait
for s in $GPUS; do grep -q saved "$OUT/logs/scan.s$s.log" || { echo "scan failed: $OUT/logs/scan.s$s.log"; exit 1; }; done
python scripts/heads/scan_heads.py report --out-dir "$OUT" | tee "$OUT/report.md"
python scripts/heads/head_pairs.py --out-dir "$OUT" | tee "$OUT/pairs.md"
