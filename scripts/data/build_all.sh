#!/bin/bash
# Build the nine evaluation candidate files and the training mixture, in dependency order
# (see docs/DATA.md). GPU stages run one shard per visible GPU; the NarrativeQA silver stage calls
# the OpenAI API and needs OPENAI_API_KEY.
#
#   bash scripts/data/build_all.sh                 # everything
#   PART=eval  bash scripts/data/build_all.sh      # evaluation sets only
#   PART=train bash scripts/data/build_all.sh      # training mixture only
#   DRY=1      bash scripts/data/build_all.sh      # print the commands
# Data root: $RE3RANK_DATA (default <repo>/data); logs in <data>/logs/.
set -u
cd "$(dirname "$0")/../.." || exit 9
D=scripts/data
PART=${PART:-all}
DRY=${DRY:-0}
NGPU=${NGPU:-$( [ "$DRY" = 1 ] && echo 1 || nvidia-smi -L | wc -l)}
LOG=${RE3RANK_DATA:-data}/logs
mkdir -p "$LOG"
export TOKENIZERS_PARALLELISM=false

run() {  # log-name command...
  local name=$1; shift
  echo "== $*"
  [ "$DRY" = 1 ] && return 0
  "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED ($name), see $LOG/$name.log"; exit 1; }
}

shards() {  # log-name nshards script stage [args...]: shard s on GPU s % NGPU, NGPU at a time
  local name=$1 n=$2 script=$3 stage=$4; shift 4
  local pids=() s
  for ((s = 0; s < n; s++)); do
    echo "== CUDA_VISIBLE_DEVICES=$((s % NGPU)) python $D/$script $stage --shard $s --nshards $n $*"
    if [ "$DRY" != 1 ]; then
      CUDA_VISIBLE_DEVICES=$((s % NGPU)) python $D/$script "$stage" --shard $s --nshards $n "$@" \
        > "$LOG/$name.s$s.log" 2>&1 &
      pids+=($!)
    fi
    if (( (s + 1) % NGPU == 0 || s + 1 == n )); then
      for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || { echo "FAILED ($name), see $LOG/$name.s*.log"; exit 1; }; done
      pids=()
    fi
  done
}

[ "$DRY" = 1 ] || bash $D/download_raw.sh

# NarrativeQA: both splits share docs.json, chunks.json and the book embeddings
nqa_prepare() {  # split...
  local sp
  for sp in "$@"; do run "nqa_prepare_$sp" python $D/build_narrativeqa.py prepare --split "$sp"; done
  run nqa_chunk python $D/build_narrativeqa.py chunk
}

if [ "$PART" = all ] || [ "$PART" = eval ]; then
  run hipporag env CUDA_VISIBLE_DEVICES=0 python $D/build_hipporag.py --dataset all
  run detqa_prep  python $D/build_detectiveqa.py prep
  run detqa_chunk python $D/build_detectiveqa.py chunk
  shards detqa_embed "$NGPU" build_detectiveqa.py embed
  run detqa_assemble python $D/build_detectiveqa.py assemble
  run locomo env CUDA_VISIBLE_DEVICES=0 python $D/build_locomo.py
  run lme_parse python $D/build_longmemeval.py parse
  shards lme_embed 4 build_longmemeval.py embed          # 4 shards: fixes the item order
  run lme_assemble python $D/build_longmemeval.py assemble
  nqa_prepare test train
  shards nqa_embed_test "$NGPU" build_narrativeqa.py embed --split test
  run nqa_silver_test   python $D/build_narrativeqa.py silver --split test
  run nqa_assemble_test python $D/build_narrativeqa.py assemble --split test
fi

if [ "$PART" = all ] || [ "$PART" = train ]; then
  [ "$PART" = train ] && nqa_prepare train
  shards nqa_embed_train "$NGPU" build_narrativeqa.py embed --split train
  run nqa_silver_train   python $D/build_narrativeqa.py silver --split train
  run nqa_assemble_train python $D/build_narrativeqa.py assemble --split train
  run musique_prep python $D/build_musique_train.py prep
  shards musique_embed 3 build_musique_train.py embed       # 3 shards, as released
  run musique_assemble env CUDA_VISIBLE_DEVICES=0 python $D/build_musique_train.py assemble --nshards 3
  run extra_parse python $D/build_qaconv_friendsqa_squad.py parse
  for src in qaconv friends squad; do
    run "extra_embed_$src" env CUDA_VISIBLE_DEVICES=0 python $D/build_qaconv_friendsqa_squad.py embed --source $src
  done
  run extra_mix python $D/build_qaconv_friendsqa_squad.py mix
  run cvm_parse python $D/build_convomem.py parse
  run cvm_split python $D/build_convomem.py split --nshards 4
  shards cvm_embed 4 build_convomem.py embed              # 4 shards: fixes the sampling order
  run cvm_assemble python $D/build_convomem.py assemble
  run train_mix python $D/build_train_mix.py
fi
echo "BUILD DONE ($PART)"
