#!/bin/bash
# Download the raw files that are not on the Hugging Face Hub into $RE3RANK_DATA/raw
# (default <repo>/data/raw), then check them against the copies the released data was built from.
#   raw/locomo/locomo10.json       LoCoMo-10             github.com/snap-research/locomo
#   raw/qaconv/QAConv-V1.1/        QAConv V1.1 (zip)     github.com/salesforce/QAConv
#   raw/friendsqa/friendsqa_*.json FriendsQA trn/dev/tst github.com/emorynlp/FriendsQA
# The Hugging Face datasets (osunlp/HippoRAG_2, deepmind/narrativeqa, Phospheneser/DetectiveQA,
# xiaowu0162/longmemeval, dgslibisey/MuSiQue, rajpurkar/squad, Salesforce/ConvoMem) are fetched by
# the build scripts themselves through the HF cache.
#
#   bash scripts/data/download_raw.sh
set -u
cd "$(dirname "$0")/../.." || exit 9
RAW=${RE3RANK_DATA:-data}/raw
mkdir -p "$RAW/locomo" "$RAW/qaconv" "$RAW/friendsqa"

fetch() {  # url dest
  [ -s "$2" ] && { echo "have $2"; return 0; }
  echo "get  $1"
  curl -sSL --fail --max-time 600 -o "$2.part" "$1" && mv "$2.part" "$2" || { echo "DOWNLOAD FAILED: $1"; rm -f "$2.part"; return 1; }
}

fetch https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json "$RAW/locomo/locomo10.json"
fetch https://raw.githubusercontent.com/salesforce/QAConv/master/dataset/QAConv-V1.1.zip "$RAW/qaconv/QAConv-V1.1.zip"
for s in trn dev tst; do
  fetch "https://raw.githubusercontent.com/emorynlp/FriendsQA/master/dat/friendsqa_$s.json" "$RAW/friendsqa/friendsqa_$s.json"
done
[ -s "$RAW/qaconv/QAConv-V1.1/trn.json" ] || unzip -q -o "$RAW/qaconv/QAConv-V1.1.zip" -d "$RAW/qaconv"

# sha256 of the copies verified to reproduce the released candidate files
check() {  # sha256 file
  [ -f "$2" ] || { echo "MISSING  $2"; return; }
  [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$1" ] && echo "ok       $2" || echo "MISMATCH $2 (upstream changed?)"
}
check 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4 "$RAW/locomo/locomo10.json"
check 74c6039a1fb12d1e2d7e64c97565a3a83321af206e043f76f1b38183921563db "$RAW/qaconv/QAConv-V1.1.zip"
check d63685693b30cee5d33b7cf2e6408cd284c05e49be1b3b06954c4d01cfe1c91b "$RAW/friendsqa/friendsqa_trn.json"
check 3a85d3fcb175589ffd050fb4bf431da77a6e12589c8ec03f1d54060ce33ec703 "$RAW/friendsqa/friendsqa_dev.json"
check a98f24d4b04ac9f8d9468088febf304bd90eada134537277c1a425e31c0372bc "$RAW/friendsqa/friendsqa_tst.json"
