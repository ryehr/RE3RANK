#!/bin/bash
# Rankings for the LoCoMo downstream QA: every one of the 1531 questions of the LoCoMo candidate file
# needs one, including the 29 with no gold chunk in the pool, so the scorers run with --order-out /
# --keep-nogold here. One system per call; the line printed at the end is the --system spec to hand
# to run_locomo_qa.sh.
#
#   QRRanker (transformers-4.53 environment, docs/BASELINES.md):
#     SYSTEM=qrranker QRRANKER_PY=venv-qrranker/bin/python bash scripts/downstream/rank_locomo.sh
#   RE3Rank (a checkpoint evaluated with scripts/eval/run_eval.sh, whose LoCoMo traces are reused):
#     SYSTEM=re3rank NAME=re3rank-4b CKPT=outputs/ckpt/re3rank-4b/final BASE=Qwen/Qwen3-4B HEADS=23-10,21-8 \
#       bash scripts/downstream/rank_locomo.sh
#   (CKPT can also be a released model on the Hugging Face Hub, e.g. CKPT=Ryehr/RE3RANK-Qwen3-4B.)
#
# For RE3Rank, TRACES defaults to $RE3RANK_OUT/eval/<run_eval.sh tag>/traces/locomo.json (TAG as in
# run_eval.sh: <run name>_<checkpoint dir>, or the released model's name). External rerankers need no extra step: their nine-set
# score file $RE3RANK_OUT/baselines/<tag>/locomo.json already covers every question.
# Outputs go to $RE3RANK_OUT/downstream/locomo_rank/ (GPU: CUDA_VISIBLE_DEVICES, default 0).
set -u
cd "$(dirname "$0")/../.." || exit 9
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
SYSTEM=${SYSTEM:?qrranker or re3rank}
OUT_ROOT=${RE3RANK_OUT:-outputs}
RD=$OUT_ROOT/downstream/locomo_rank; mkdir -p "$RD"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
case $SYSTEM in
  qrranker)
    PY=${QRRANKER_PY:?python of the transformers-4.53 environment, e.g. venv-qrranker/bin/python}
    NAME=${NAME:-qrranker}
    $PY scripts/baselines/qrranker_score.py --data locomo --shard 0 --nshards 1 \
      --out "$RD/$NAME.s0.json" --order-out "$RD/$NAME.order.json" > "$RD/$NAME.log" 2>&1
    [ -s "$RD/$NAME.order.json" ] || { echo "FAILED: $NAME (see $RD/$NAME.log)"; exit 1; }
    grep -A1 conditional "$RD/$NAME.log" | tail -1
    echo "$NAME=order:$RD/$NAME.order.json"
    ;;
  re3rank)
    NAME=${NAME:?system name, e.g. re3rank-4b}
    CKPT=${CKPT:?checkpoint dir holding the scorer, or a released model}
    BASE=${BASE:?backbone, e.g. Qwen/Qwen3-4B}
    HEADS=${HEADS:?scoring heads, e.g. 23-10,21-8}
    if [ ! -e "$CKPT" ]; then         # a released model on the Hub: use its local snapshot
      LOCAL=$(python -m re3rank.hub "$CKPT") || exit 1
      [ "$LOCAL" != "$CKPT" ] && { TAG=${TAG:-$(basename "$CKPT")}; CKPT=$LOCAL; }
    fi
    TAG=${TAG:-$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")}
    TRACES=${TRACES:-$OUT_ROOT/eval/$TAG/traces/locomo.json}
    [ -s "$TRACES" ] || { echo "no traces at $TRACES: run scripts/eval/run_eval.sh on this checkpoint first (SETS=locomo is enough)"; exit 1; }
    SC=""; for f in "$CKPT/scorer/scorer.safetensors" "$CKPT/scorer.pt"; do [ -f "$f" ] && { SC=$f; break; }; done
    python scripts/eval/score.py --model "$BASE" ${SC:+--ckpt "$SC"} --heads "$HEADS" --traces "$TRACES" \
      --data locomo --out "$RD/$NAME.json" --lowmem --keep-nogold > "$RD/$NAME.log" 2>&1
    grep -q wrote "$RD/$NAME.log" || { echo "FAILED: $NAME (see $RD/$NAME.log)"; exit 1; }
    grep -A1 conditional "$RD/$NAME.log" | tail -1
    echo "$NAME=scores:$RD/$NAME.json"
    ;;
  *) echo "SYSTEM must be qrranker or re3rank"; exit 2 ;;
esac
