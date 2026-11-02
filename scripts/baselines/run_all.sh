#!/bin/bash
# Every reranking baseline of the paper on the nine sets, cheapest model first, then the macro-9 table.
# Same first stage as every RE3Rank run (the top-50 already in the candidate files).
#
#   QRRANKER_PY=venv-qrranker/bin/python bash scripts/baselines/run_all.sh
#
# Without QRRANKER_PY the QRRanker row is skipped (it needs the transformers-4.53 environment,
# docs/BASELINES.md). Each step skips the sets it has already scored, so the script can be rerun.
# The table goes to $RE3RANK_OUT/baselines/summary.md; add RE3Rank result directories to it with
# summary.py (e.g. outputs/eval/<tag>/scores=RE3Rank-4B).
set -u
cd "$(dirname "$0")/../.." || exit 9
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT_ROOT=${RE3RANK_OUT:-outputs}

# reference row: no reranking at all, the retriever's own order
FD=$OUT_ROOT/baselines/first-stage; mkdir -p "$FD"
for st in hotpotqa 2wiki musique nq popqa narrativeqa detectiveqa locomo longmemeval; do
  dn=""; [ "$st" = hotpotqa ] && dn="--gold-denom 2"
  [ -s "$FD/$st.txt" ] || python scripts/baselines/eval_scores.py --data "$st" --first-stage $dn > "$FD/$st.txt" 2>&1
done
echo "######## first-stage reference done  $(date +%m-%d\ %H:%M:%S)"

# tag  model  method
for spec in "jina-reranker-v3.5 jinaai/jina-reranker-v3.5 jina" \
            "qwen3-reranker-0.6b Qwen/Qwen3-Reranker-0.6B qwen3" \
            "zerank-2 zeroentropy/zerank-2-reranker zerank" \
            "qwen3-reranker-4b Qwen/Qwen3-Reranker-4B qwen3" \
            "qwen3-reranker-8b Qwen/Qwen3-Reranker-8B qwen3" \
            "rearank-7b le723z/Rearank-7B rearank" \
            "reasonrank-7b liuwenhan/reasonrank-7B reasonrank" \
            "grouprank-7b AQ-MedAI/Diver-GroupRank-7B grouprank" \
            "grouprank-32b AQ-MedAI/Diver-GroupRank-32B grouprank" \
            "reasonrank-32b liuwenhan/reasonrank-32B reasonrank"; do
  set -- $spec
  echo "######## $1 ($3, $2)  $(date +%m-%d\ %H:%M:%S)"
  TAG=$1 MODEL=$2 METHOD=$3 bash scripts/baselines/run_baseline.sh 2>&1 | grep -v "^merged\|^####"
done

if [ -n "${QRRANKER_PY:-}" ]; then
  echo "######## qrranker (MindscapeRAG/QRRanker)  $(date +%m-%d\ %H:%M:%S)"
  bash scripts/baselines/run_qrranker.sh 2>&1 | grep -v "^####"
else
  echo "######## QRRANKER_PY not set: QRRanker skipped"
fi

python scripts/baselines/summary.py first-stage=First-stage qrranker=QRRanker \
  qwen3-reranker-0.6b="Qwen3-Reranker (0.6B)" qwen3-reranker-4b="Qwen3-Reranker (4B)" \
  qwen3-reranker-8b="Qwen3-Reranker (8B)" zerank-2="zerank-2 (4B)" \
  jina-reranker-v3.5="jina-reranker-v3.5 (0.6B)" grouprank-7b="GroupRank (7B)" grouprank-32b="GroupRank (32B)" \
  rearank-7b="REARANK (7B)" reasonrank-7b="ReasonRank (7B)" reasonrank-32b="ReasonRank (32B)" \
  | tee "$OUT_ROOT/baselines/summary.md"
echo "ALL BASELINES DONE  $(date +%m-%d\ %H:%M:%S)"
