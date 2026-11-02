#!/bin/bash
# The released QRRanker (MindscapeRAG/QRRanker: 16 trained QR heads, 25-layer truncated
# Qwen3-4B-Instruct-2507) on the nine sets, through qrranker_score.py, which drives it with the
# RE3Rank prompt construction (re3rank.prompts.build_prompt) and the same conditional / NDCG
# conventions, so the row is comparable with the RE3Rank scorer rather than with the paper's protocol.
#
#   QRRANKER_PY=venv-qrranker/bin/python bash scripts/baselines/run_qrranker.sh
#   smoke: add LIMIT=40 SETS="locomo"
#
# The checkpoint's custom modeling code targets the transformers ~4.53 cache API
# (DynamicCacheWithQuery needs _seen_tokens, dropped in 4.54; the rope registry and config field
# names moved again in 5.x), so it runs in its own environment: QRRANKER_PY is that environment's
# python (docs/BASELINES.md: transformers 4.53.0, huggingface_hub 0.36.2, kernels 0.9.0, torch from
# the main environment via --system-site-packages; a newer kernels package makes transformers 4.53
# fail to import). Per set, under $RE3RANK_OUT/baselines/$TAG: <set>.s<gpu>.json (per-item records,
# also read by sig_test.py 'official:'), <set>.txt (metrics), logs/.
set -u
cd "$(dirname "$0")/../.." || exit 9
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
PY=${QRRANKER_PY:?python of the transformers-4.53 environment, e.g. venv-qrranker/bin/python}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}; GPUS=$(seq -s " " 0 $((NGPU-1)))
TAG=${TAG:-qrranker}
LIMIT=${LIMIT:-0}; LIM=""; [ "$LIMIT" -gt 0 ] && LIM="--limit $LIMIT"
OUT_ROOT=${RE3RANK_OUT:-outputs}
SD=$OUT_ROOT/baselines/$TAG; LD=$SD/logs; mkdir -p "$LD"
SETS=${SETS:-"narrativeqa locomo hotpotqa 2wiki musique nq popqa detectiveqa longmemeval"}
for st in $SETS; do
  dn=""; [ "$st" = hotpotqa ] && dn="--gold-denom 2"
  if [ ! -s "$SD/$st.s0.json" ]; then
    echo "######## $TAG $st  $(date +%m-%d\ %H:%M:%S)"
    for g in $GPUS; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/baselines/qrranker_score.py --data "$st" $dn $LIM \
        --shard "$g" --nshards "$NGPU" --out "$SD/$st.s$g.json" > "$LD/$st.s$g.log" 2>&1 &
    done
    wait
  fi
  $PY scripts/baselines/qrranker_score.py --report "$SD/$st".s*.json > "$SD/$st.txt" 2>&1
  grep -q conditional "$SD/$st.txt" \
    && { echo "$TAG $st done"; grep -A1 conditional "$SD/$st.txt" | tail -1; } \
    || echo "FAILED: $TAG $st (see $SD/$st.txt and $LD/)"
done
echo "QRRANKER DONE  $(date +%m-%d\ %H:%M:%S)"
