"""Rank heads, and pairs of heads, by the conditional R@3 / NDCG@10 they reach without training,
from a scan directory of scan_heads.py (RR masses: query = reasoning trace).

A head ranks the 50 candidates of an item by its attention mass; a pair by the sum of its two heads'
masses (the same order as the scorer's mean over heads). Ties keep the candidate order. Pairs are
searched among the top-N single heads.

  python scripts/heads/head_pairs.py --out-dir D [--top 30] [--report 23-10,21-8 ...]
"""
import argparse
import itertools

import numpy as np

from scan_heads import load_scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top", type=int, default=30, help="pairs are searched among the top-N single heads")
    ap.add_argument("--report", nargs="*", default=[], help="head sets to report, e.g. 23-10,21-8")
    args = ap.parse_args()

    _, M, G, _ = load_scan(args.out_dir)
    N, L, H, C = M.shape
    keep = G.sum(1) > 0
    M, G = M[keep], G[keep]
    n_in = G.sum(1)
    disc = 1.0 / np.log2(np.arange(2, 12))
    idcg = np.array([disc[:min(10, k)].sum() for k in n_in])

    def metrics(mass):                     # (N, C) -> conditional R@3, NDCG@10 in percent
        order = np.lexsort((np.arange(C)[None, :].repeat(len(mass), 0), -mass), axis=1)
        g = np.take_along_axis(G, order, 1).astype(np.float32)
        return 100 * (g[:, :3].sum(1) / n_in).mean(), 100 * ((g[:, :10] * disc[None, :]).sum(1) / idcg).mean()

    flat = M.reshape(len(M), L * H, C)
    name = lambda i: f"{i // H}-{i % H}"
    single = sorted(((metrics(flat[:, i]), i) for i in range(L * H)), key=lambda x: -x[0][0])
    print(f"{len(M)} items, {L}x{H} heads\n\n### single heads by R@3\n| rank | head | R@3 | NDCG@10 |\n|---|---|---|---|")
    for r, ((r3, nd), i) in enumerate(single[:12], 1):
        print(f"| {r} | {name(i)} | {r3:.2f} | {nd:.2f} |")
    cand = [i for _, i in single[:args.top]]
    pairs = sorted(((*metrics(flat[:, a] + flat[:, b]), a, b) for a, b in itertools.combinations(cand, 2)),
                   key=lambda x: -x[0])
    print(f"\n### pairs by R@3 (among the top-{args.top} single heads)\n| rank | pair | R@3 | NDCG@10 |\n|---|---|---|---|")
    for r, (r3, nd, a, b) in enumerate(pairs[:10], 1):
        print(f"| {r} | {name(a)},{name(b)} | {r3:.2f} | {nd:.2f} |")
    print(f"\nBEST_PAIR {name(pairs[0][2])},{name(pairs[0][3])}")
    for spec in args.report:
        idx = [int(h.split("-")[0]) * H + int(h.split("-")[1]) for h in spec.split(",")]
        r3, nd = metrics(sum(flat[:, i] for i in idx))
        print(f"{spec}: R@3 {r3:.2f}  NDCG@10 {nd:.2f}")


if __name__ == "__main__":
    main()
