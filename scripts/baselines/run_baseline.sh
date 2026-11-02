#!/bin/bash
# Nine-set evaluation of one external reranker (rerank_score.py) on the SAME first stage as every
# RE3Rank run: the top-50 candidates already in the candidate files, so only the reranking differs.
#
#   TAG=qwen3-reranker-4b MODEL=Qwen/Qwen3-Reranker-4B METHOD=qwen3 bash scripts/baselines/run_baseline.sh
#   smoke: add LIMIT=40 SETS="locomo"
#
# METHOD: qwen3 | jina | zerank | grouprank | rearank | reasonrank (rerank_score.py --help; every model id
# and its method are listed in run_all.sh). One shard per GPU, then merged. Per set, under
# $RE3RANK_OUT/baselines/$TAG: <set>.json (scores {item id: [score per candidate]}), <set>.txt (metrics,
# eval_scores.py), logs/<set>.s<gpu>.log, and raw/<set>.s<gpu>.jsonl (every window reply of the listwise
# reasoning rerankers). A set whose merged scores exist is only re-evaluated.
set -u
cd "$(dirname "$0")/../.." || exit 9
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
TAG=${TAG:?short name of the run, e.g. qwen3-reranker-4b}
MODEL=${MODEL:?hub id or path}
METHOD=${METHOD:?qwen3|jina|zerank|grouprank|rearank|reasonrank}
LIMIT=${LIMIT:-0}
GPU_MEM=${GPU_MEM:-0.85}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}; GPUS=$(seq -s " " 0 $((NGPU-1)))
OUT_ROOT=${RE3RANK_OUT:-outputs}
SD=$OUT_ROOT/baselines/$TAG; LD=$SD/logs; mkdir -p "$LD"
# longest sets first
SETS=${SETS:-"narrativeqa locomo hotpotqa 2wiki musique nq popqa detectiveqa longmemeval"}
LIM=""; [ "$LIMIT" -gt 0 ] && LIM="--limit $LIMIT"
for st in $SETS; do
  out=$SD/$st.json
  if [ ! -s "$out" ]; then
    echo "######## $TAG $st scores ($METHOD)  $(date +%m-%d\ %H:%M:%S)"
    for g in $GPUS; do
      CUDA_VISIBLE_DEVICES=$g python scripts/baselines/rerank_score.py --data "$st" \
        --method "$METHOD" --model "$MODEL" --out "$out" --shard "$g" --nshards "$NGPU" \
        --gpu-mem "$GPU_MEM" $LIM > "$LD/$st.s$g.log" 2>&1 &
    done
    wait
    python scripts/baselines/rerank_score.py --data "$st" --method "$METHOD" --model "$MODEL" \
      --out "$out" --nshards "$NGPU" --merge || { echo "MERGE FAILED: $TAG $st"; continue; }
  fi
  dn=""; [ "$st" = hotpotqa ] && dn="--gold-denom 2"
  python scripts/baselines/eval_scores.py --data "$st" --scores "$out" $dn $LIM > "$SD/$st.txt" 2>&1
  grep -q conditional "$SD/$st.txt" \
    && { echo "$TAG $st done"; grep -A1 conditional "$SD/$st.txt" | tail -1; } \
    || echo "EVAL FAILED: $TAG $st (see $SD/$st.txt)"
done
echo "BASELINE DONE $TAG  $(date +%m-%d\ %H:%M:%S)"
