"""Conditional ranking metrics.

Only queries with at least one gold candidate among the 50 are scored, and R@k divides by the number
of gold candidates in the pool: R@k = |gold in top-k| / |gold in pool|. NDCG@10 uses binary relevance
with the ideal DCG over min(10, |gold in pool|). Ties keep the first-stage order.
"""
import math

import torch

KS = (1, 3, 5, 10)


def query_metrics(scores, gold, ks=KS):
    """({k: R@k}, NDCG@10) of one query; gold is a bool tensor with at least one True."""
    n_in = int(gold.sum().item())
    order = torch.argsort(scores, descending=True).cpu()
    dcg = sum(float(gold[order[i]]) / math.log2(i + 2) for i in range(min(10, len(order))))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, n_in)))
    return {k: int(gold[order[:k]].sum().item()) / n_in for k in ks}, dcg / idcg


def summary_line(recall, ndcg):
    """The line every evaluation prints (and run_eval.sh parses):
      R@1=..  R@3=..  R@5=..  R@10=..  NDCG@10=..   (percent, averaged over queries)"""
    return ("  " + "  ".join(f"R@{k}={100 * sum(v) / len(v):.2f}" for k, v in recall.items())
            + f"  NDCG@10={100 * sum(ndcg) / len(ndcg):.2f}")
