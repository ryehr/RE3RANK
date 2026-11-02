#!/bin/bash
# Train RE3Rank (one epoch over the training mixture) with the paper's settings, resuming from the
# newest checkpoint in the run directory if there is one.
#
#   SIZE=4b   bash scripts/train/run_train.sh      # Qwen3-4B,  heads 23-10,21-8,  8 GPUs
#   SIZE=0.6b bash scripts/train/run_train.sh      # Qwen3-0.6B, heads 20-4,16-8,  8 GPUs
#   SIZE=8b   bash scripts/train/run_train.sh      # Qwen3-8B,  heads 15-11,23-10, 8 GPUs (2 per rank)
#
# Every update consumes 96 items (8 ranks x 12, or 4 ranks x 24 at 8B); the 37,759 usable training
# items make one epoch of 394 updates. Override BASE / HEADS / NPROC / EPR / TWO_GPU / RUN to deviate.
set -u
cd "$(dirname "$0")/../.." || exit 9
SIZE=${SIZE:-4b}
case $SIZE in
  0.6b) D_BASE=Qwen/Qwen3-0.6B; D_HEADS=20-4,16-8;  D_NPROC=8; D_EPR=12; D_TWO=0 ;;
  4b)   D_BASE=Qwen/Qwen3-4B;   D_HEADS=23-10,21-8; D_NPROC=8; D_EPR=12; D_TWO=0 ;;
  8b)   D_BASE=Qwen/Qwen3-8B;   D_HEADS=15-11,23-10; D_NPROC=4; D_EPR=24; D_TWO=1 ;;
  *) echo "SIZE must be 0.6b, 4b or 8b"; exit 2 ;;
esac
BASE=${BASE:-$D_BASE}; HEADS=${HEADS:-$D_HEADS}; NPROC=${NPROC:-$D_NPROC}; EPR=${EPR:-$D_EPR}
TWO_GPU=${TWO_GPU:-$D_TWO}
RUN=${RUN:-re3rank-$SIZE}
OUT=${RE3RANK_OUT:-outputs}/ckpt/$RUN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
for attempt in 1 2 3; do
  [ -f "$OUT/final/SAVED" ] && { echo "$RUN: final checkpoint present"; break; }
  LATEST=$( (ls -d "$OUT"/upd* 2>/dev/null | while read -r d; do [ -f "$d/SAVED" ] && echo "${d##*upd}"; done; echo 0) | sort -n | tail -1)
  INIT=$BASE; [ "$LATEST" -gt 0 ] && INIT=$OUT/upd$LATEST
  echo "######## $RUN ($BASE, heads $HEADS) attempt $attempt: from update $LATEST  $(date +%m-%d\ %H:%M:%S)"
  torchrun --nproc_per_node "$NPROC" --master_port ${MASTER_PORT:-29708} scripts/train/train.py \
    --base "$BASE" --heads "$HEADS" --out "$OUT" --init "$INIT" --upd-offset "$LATEST" \
    --eps-per-rank "$EPR" --gen-batch "$EPR" $( [ "$TWO_GPU" = 1 ] && echo --two-gpu ) "$@"
  [ $? -eq 0 ] && break
  sleep 30
done
