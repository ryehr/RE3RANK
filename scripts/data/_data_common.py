"""Shared constants and helpers of the dataset builders in scripts/data/.

Layout under the data root (paths.DATA, env RE3RANK_DATA):
  raw/<source>/...         files downloaded by scripts/data/download_raw.sh (not on the HF Hub)
  work/<source>/...        intermediate files of the multi-stage builders
  candidates/<set>.json    the nine evaluation candidate files (paths.EVAL_SETS) and the
                           training mixture (paths.TRAIN_MIX)
Hugging Face datasets are read through the HF cache (HF_HOME) by hub id, pinned to the revision
the released files were built from.
"""
import json
import os

from re3rank import paths

# Tokenizer used for every token count, chunk budget and message cap (Qwen3 vocabulary).
TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
# First-stage retriever: candidates are its cosine top-50, in retrieval order.
EMB_ID = "Qwen/Qwen3-Embedding-8B"
TOPK = 50


def raw(*parts):
    return paths.data("raw", *parts)


def work(*parts):
    return paths.data("work", *parts)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def load_embedder(max_seq_length=None):
    """Qwen3-Embedding-8B in bf16 with left padding; max_seq_length caps the input length for the
    builders that set it (dialogue windows)."""
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        EMB_ID, model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return model


def hf_snapshot(repo_id, revision, allow_patterns=None):
    """Local path of a Hugging Face dataset snapshot (downloaded into the HF cache if missing)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, repo_type="dataset", revision=revision,
                             allow_patterns=allow_patterns)


def hf_file(repo_id, filename, revision):
    """Local path of one file of a Hugging Face dataset (downloaded into the HF cache if missing)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)
