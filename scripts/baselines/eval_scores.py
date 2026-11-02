"""Metrics for an external reranker's scores, the same definitions as scripts/eval/score.py.

conditional: R@k = hits@k / (golds present in the candidate pool), NDCG@10 over binary
relevance with IDCG over min(10, golds-in-pool); only queries with a gold in the pool count.
absolute: R@k = hits@k / n_gold (or --gold-denom). Ties keep the first-stage order, which
matters for the groupwise baselines: their scores are integers 0-10 averaged over two epochs,
so ties are common (the RE3Rank scorer's are continuous). Printed in score.py's format so the
same summary table reads both.

  python scripts/baselines/eval_scores.py --data musique --scores outputs/baselines/jina-reranker-v3.5/musique.json
  python scripts/baselines/eval_scores.py --data hotpotqa --first-stage --gold-denom 2   # no-reranking row
"""
import argparse
import json
import math

from re3rank import paths

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="candidate file, or the name of an evaluation set")
ap.add_argument("--scores", default="", help="{item id: [score per candidate]} (rerank_score.py)")
ap.add_argument("--first-stage", action="store_true",
                help="no scores: rank by the candidate file's own (first-stage) order")
ap.add_argument("--gold-denom", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
if bool(a.scores) == a.first_stage:
    ap.error("give exactly one of --scores and --first-stage")

data = json.load(open(paths.eval_file(a.data) if a.data in paths.EVAL_SETS else a.data))
if a.limit:
    data = data[: a.limit]
if a.first_stage:
    S = {str(item["id"]): [-j for j in range(len(item["paragraphs"]))] for item in data}
else:
    S = json.load(open(a.scores))
ks = (1, 3, 5, 10)
cond = {k: [] for k in ks}
absr = {k: [] for k in ks}
nd10 = []
missing = 0
for item in data:
    gold = [bool(p["is_supporting"]) for p in item["paragraphs"]]
    n_in = sum(gold)
    denom_abs = a.gold_denom or item.get("n_gold", 0)
    if n_in == 0:
        if denom_abs:
            for k in ks:
                absr[k].append(0.0)
        continue
    s = S.get(str(item["id"]))
    if s is None or len(s) != len(gold):
        missing += 1
        s = [0.0] * len(gold)
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))
    dcg = sum(float(gold[order[i]]) / math.log2(i + 2) for i in range(min(10, len(order))))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, n_in)))
    nd10.append(dcg / idcg)
    for k in ks:
        hit = sum(1 for i in order[:k] if gold[i])
        cond[k].append(hit / n_in)
        if denom_abs:
            absr[k].append(hit / denom_abs)

if missing:
    print(f"WARNING: {missing} items had no scores (ranked as given)")
print("\n=== conditional (denom = golds in candidates) ===")
print("  " + "  ".join(f"R@{k}={100*sum(v)/len(v):.2f}" for k, v in cond.items())
      + f"  NDCG@10={100*sum(nd10)/len(nd10):.2f}")
if any(absr[3]):
    print("=== absolute (paper convention) ===")
    print("  " + "  ".join(f"R@{k}={100*sum(v)/len(v):.2f}" for k, v in absr.items()))
