#!/bin/bash
# Evaluate a RE3Rank checkpoint on the nine evaluation sets.
#   1. traces: the checkpoint's generator writes a reasoning trace for every item (vLLM, greedy,
#      one shard per GPU)
#   2. scores: the checkpoint's scorer ranks the 50 candidates of every item by the attention its
#      heads pay from the trace to each candidate (MAXPER jobs per GPU)
#   3. a table of conditional R@3 / NDCG@10 per set and their macro average
#
#   CKPT=outputs/ckpt/re3rank-4b/final BASE=Qwen/Qwen3-4B HEADS=23-10,21-8 bash scripts/eval/run_eval.sh
#   CKPT=Ryehr/RE3RANK-Qwen3-4B BASE=Qwen/Qwen3-4B HEADS=23-10,21-8 bash scripts/eval/run_eval.sh
#
# CKPT holds the generator weights and the scorer: a checkpoint of scripts/train/train.py (scorer.pt),
# or a released model on the Hugging Face Hub (scorer/, downloaded on first use). An untrained
# baseline: CKPT=Qwen/Qwen3-4B (no scorer -> the untrained scorer). Results go to
# $RE3RANK_OUT/eval/$TAG (TAG defaults to the checkpoint's run name, or the released model's name).
set -u
cd "$(dirname "$0")/../.." || exit 9
CKPT=${CKPT:?generator dir with its scorer, a released model, or a hub id for the untrained baseline}
BASE=${BASE:?backbone, e.g. Qwen/Qwen3-4B}
HEADS=${HEADS:?scoring heads, e.g. 23-10,21-8}
if [ ! -e "$CKPT" ]; then           # a released model on the Hub: use its local snapshot
  LOCAL=$(python -m re3rank.hub "$CKPT") || exit 1
  [ "$LOCAL" != "$CKPT" ] && { TAG=${TAG:-$(basename "$CKPT")}; CKPT=$LOCAL; }
fi
OUT_ROOT=${RE3RANK_OUT:-outputs}
TAG=${TAG:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}
SETS=${SETS:-"hotpotqa 2wiki musique nq popqa narrativeqa detectiveqa locomo longmemeval"}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}; GPUS=$(seq -s " " 0 $((NGPU-1)))
MAXPER=${MAXPER:-3}
export TOKENIZERS_PARALLELISM=false
SC=""; for f in "$CKPT/scorer/scorer.safetensors" "$CKPT/scorer.pt"; do [ -f "$f" ] && { SC=$f; break; }; done
TD=$OUT_ROOT/eval/$TAG/traces; RD=$OUT_ROOT/eval/$TAG/scores; LD=$OUT_ROOT/eval/$TAG/logs
mkdir -p "$TD" "$RD" "$LD"

echo "######## $TAG: traces  $(date +%H:%M:%S)"
for st in $SETS; do
  [ -s "$TD/$st.json" ] && continue
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g python scripts/eval/generate_traces.py --data "$st" \
      --out "$TD/$st.s$g.json" --model "$CKPT" --tokenizer "$BASE" \
      --shard $g --nshards $NGPU > "$LD/gen.$st.s$g.log" 2>&1 &
  done
  wait
  for g in $GPUS; do grep -q wrote "$LD/gen.$st.s$g.log" || { echo "trace generation failed: $LD/gen.$st.s$g.log"; exit 1; }; done
  python scripts/eval/generate_traces.py --merge $(for g in $GPUS; do echo "$TD/$st.s$g.json"; done) --out "$TD/$st.json"
  rm -f "$TD/$st".s*.json
done

echo "######## $TAG: scores  $(date +%H:%M:%S)"
run_one() {  # gpu set
  local g=$1 st=$2 f=$RD/$2.txt dn=""
  [ -s "$f" ] && grep -q conditional "$f" && return 0
  [ "$st" = hotpotqa ] && dn="--gold-denom 2"
  CUDA_VISIBLE_DEVICES=$g python scripts/eval/score.py --model "$BASE" ${SC:+--ckpt "$SC"} \
    --heads "$HEADS" --traces "$TD/$st.json" --data "$st" $dn --out "$RD/$st.json" > "$f" 2>&1
  grep -q conditional "$f" && echo "  $st done" || echo "  $st FAILED (see $f)"
}
JOBS=($SETS)
for g in $GPUS; do
  (
    for (( i=g; i<${#JOBS[@]}; i+=NGPU )); do
      while [ "$(jobs -rp | wc -l)" -ge "$MAXPER" ]; do sleep 10; done
      run_one $g ${JOBS[$i]} &
      sleep 20
    done
    wait
  ) &
done
wait

python - "$RD" $SETS <<'PY' | tee "$OUT_ROOT/eval/$TAG/summary.md"
import re, sys
rd, sets = sys.argv[1], sys.argv[2:]
print("| set | R@1 | R@3 | R@5 | NDCG@10 |\n|---|---|---|---|---|")
cols = []
for st in sets:
    try:
        t = open(f"{rd}/{st}.txt").read()
        v = [float(re.search(rf"{m}=([\d.]+)", t).group(1)) for m in ("R@1", "R@3", "R@5", "NDCG@10")]
        cols.append(v)
        print(f"| {st} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    except (OSError, AttributeError):
        print(f"| {st} | -- | -- | -- | -- |")
if len(cols) == len(sets):
    print(f"| **macro-{len(sets)}** | " + " | ".join(f"**{sum(c[i] for c in cols)/len(cols):.2f}**" for i in range(4)) + " |")
PY
