"""Where data and outputs live.

Two roots, set by environment variables:
  RE3RANK_DATA  datasets: raw downloads and the built candidate files   (default: <repo>/data)
  RE3RANK_OUT   checkpoints, reasoning traces and evaluation results    (default: <repo>/outputs)
Model weights come from the Hugging Face cache (HF_HOME).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("RE3RANK_DATA", os.path.join(REPO, "data"))
OUT = os.environ.get("RE3RANK_OUT", os.path.join(REPO, "outputs"))

# The nine evaluation sets: name -> candidate file under DATA (built by scripts/data/).
EVAL_SETS = {
    "hotpotqa": "candidates/hotpotqa.json",
    "2wiki": "candidates/2wiki.json",
    "musique": "candidates/musique.json",
    "nq": "candidates/nq.json",
    "popqa": "candidates/popqa.json",
    "narrativeqa": "candidates/narrativeqa.json",
    "detectiveqa": "candidates/detectiveqa.json",
    "locomo": "candidates/locomo.json",
    "longmemeval": "candidates/longmemeval.json",
}
# The training mixture (scripts/data/build_train_mix.py).
TRAIN_MIX = "candidates/train_mix.json"


def data(*parts):
    return os.path.join(DATA, *parts)


def out(*parts):
    return os.path.join(OUT, *parts)


def eval_file(name):
    return data(EVAL_SETS[name])
