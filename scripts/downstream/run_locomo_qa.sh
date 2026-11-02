#!/bin/bash
# LoCoMo downstream QA, the QRRanker paper's Table 3 protocol (locomo_qa.py): the top-3 reranked chunks
# of the Qwen3-Embedding-8B top-50 pool -> an OpenAI model -> per-category token F1. One process per
# LLM, the systems in turn; replies are cached and the run resumes where it stopped. Then the F1 tables
# (summary.md) and the paired per-question test against BASE (sig_vs_<BASE>.md), both in
# $RE3RANK_OUT/downstream/locomo_qa/.
#
#   export OPENAI_API_KEY=...
#   bash scripts/downstream/run_locomo_qa.sh first=first \
#       qrranker=order:outputs/downstream/locomo_rank/qrranker.order.json \
#       re3rank-4b=scores:outputs/downstream/locomo_rank/re3rank-4b.json
#
# Each argument is NAME=SPEC (locomo_qa.py --help): first | scores:<file> | order:<file>; the
# rankings come from rank_locomo.sh, or for an external reranker from its nine-set score file
# ($RE3RANK_OUT/baselines/<tag>/locomo.json). LLMS: the answer models (default: the paper's gpt-4o-mini gpt-5-mini gpt-6-luna);
# BASE: the system the significance test pairs every other system against (default qrranker).
set -u
cd "$(dirname "$0")/../.." || exit 9
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
[ $# -gt 0 ] || { sed -n '2,16p' "$0"; exit 2; }
[ -n "${OPENAI_API_KEY:-}" ] || { echo "set OPENAI_API_KEY"; exit 1; }
LLMS=${LLMS:-"gpt-4o-mini gpt-5-mini gpt-6-luna"}
BASE=${BASE:-qrranker}
WORKERS=${WORKERS:-12}
OUT_ROOT=${RE3RANK_OUT:-outputs}
LD=$OUT_ROOT/downstream/locomo_qa/logs; mkdir -p "$LD"
SYS=(); for s in "$@"; do SYS+=(--system "$s"); done
echo "######## LoCoMo QA ($*)  $(date +%m-%d\ %H:%M:%S)"
for llm in $LLMS; do
  ( python scripts/downstream/locomo_qa.py run --llm "$llm" "${SYS[@]}" --workers "$WORKERS" >> "$LD/$llm.log" 2>&1 \
      && echo "QA DONE $llm  $(date +%H:%M:%S)" || echo "QA FAILED $llm (see $LD/$llm.log)  $(date +%H:%M:%S)" ) &
done
wait
python scripts/downstream/locomo_qa.py report
python scripts/downstream/locomo_qa.py sig --base "$BASE"
echo "LOCOMO QA DONE  $(date +%m-%d\ %H:%M:%S)"
