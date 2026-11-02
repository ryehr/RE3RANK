"""Released RE3Rank models on the Hugging Face Hub.

A released model repo holds the generator at its root (a standard transformers / vLLM checkpoint) and
the trained scorer under scorer/: scorer.safetensors and config.json (base model, heads, and the frozen
tables that the loader takes from the base model).

  python -m re3rank.hub <ckpt>      prints the local directory to use for <ckpt>
"""
import os
import sys

from huggingface_hub import file_exists, snapshot_download

SCORER_FILES = ("scorer/scorer.safetensors", "scorer.pt")     # released model, training run


def local_checkpoint(ckpt):
    """ckpt itself if it is a local path; the local snapshot of a released RE3Rank repo (one with
    scorer/config.json), downloaded if needed; otherwise ckpt unchanged (e.g. a base model id for the
    untrained baseline)."""
    if os.path.exists(ckpt):
        return ckpt
    try:
        released = file_exists(ckpt, "scorer/config.json")
    except Exception:                  # not a valid repo id
        return ckpt
    return snapshot_download(ckpt) if released else ckpt


def scorer_file(ckpt_dir):
    """The trained scorer weights inside a checkpoint directory, or "" (untrained)."""
    for f in SCORER_FILES:
        if os.path.isfile(os.path.join(ckpt_dir, f)):
            return os.path.join(ckpt_dir, f)
    return ""


if __name__ == "__main__":
    print(local_checkpoint(sys.argv[1]))
