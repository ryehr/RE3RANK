#!/bin/bash
# Download every baseline reranker into the Hugging Face cache ($HF_HOME). A gated or rate-limited
# download picks up a token from HF_TOKEN or from `hf auth login`.
#
#   bash scripts/baselines/download.sh
set -u
for m in MindscapeRAG/QRRanker \
         Qwen/Qwen3-Reranker-0.6B Qwen/Qwen3-Reranker-4B Qwen/Qwen3-Reranker-8B \
         jinaai/jina-reranker-v3.5 zeroentropy/zerank-2-reranker \
         AQ-MedAI/Diver-GroupRank-7B AQ-MedAI/Diver-GroupRank-32B \
         le723z/Rearank-7B liuwenhan/reasonrank-7B liuwenhan/reasonrank-32B; do
  echo "######## download $m  $(date +%m-%d\ %H:%M:%S)"
  hf download "$m" --quiet || { echo "DOWNLOAD FAILED: $m"; continue; }
done
echo "DOWNLOADS DONE  $(date +%m-%d\ %H:%M:%S)"
