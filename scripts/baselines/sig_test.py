"""Paired per-query significance tests between two rerankers on the same candidate pools.

Per query (only queries with a gold in the 50 candidates, i.e. the conditional-metric set):
R@1, R@3, R@5 (hits@k / golds-in-pool) and NDCG@10 (binary, IDCG over min(10, golds-in-pool)),
exactly the definitions behind the tables (scripts/eval/score.py, eval_scores.py).

Tests on the per-query differences d = A - B:
  * two-sided paired randomization (sign-flip) test, B permutations (primary; no normality
    assumption, the standard IR choice for discrete per-query metrics)
  * two-sided paired t-test (secondary check)
  * 95% bootstrap CI of the mean difference (in points)
Markers: *** p<0.001, ** p<0.01, * p<0.05 (randomization p).

Per-query sources:
  scores:<dir>     <dir>/<set>.json {id: [50 scores]} (rerank_score.py, e.g. $RE3RANK_OUT/baselines/<tag>),
                   ties keep the first-stage order
  official:<glob>  qrranker_score.py --out shard dumps (records in data[shard::nshards] order); '{st}' in the
                   glob stands for the set, e.g. 'official:outputs/baselines/qrranker/{st}.s*.json'
  dump:<dir>       <dir>/<set>.json written by scripts/eval/score.py --out {id: {"s": scores, "m": metrics}}
                   (the RE3Rank scorer, e.g. $RE3RANK_OUT/eval/<tag>/scores; written without --keep-nogold)

  python scripts/baselines/sig_test.py musique dump:outputs/eval/re3rank-4b_final/scores \\
      scores:outputs/baselines/jina-reranker-v3.5
"""
import glob, json, math, re, sys
import numpy as np

from re3rank import paths

SETS = {name: paths.eval_file(name) for name in paths.EVAL_SETS}
METRICS = ("R@1", "R@3", "R@5", "NDCG@10")
DISC = 1.0 / np.log2(np.arange(2, 12))


def _row(gold_sorted, n_in):
    g = np.asarray(gold_sorted, dtype=np.float64)
    idcg = DISC[:min(10, n_in)].sum()
    return [g[:1].sum() / n_in, g[:3].sum() / n_in, g[:5].sum() / n_in, (g[:10] * DISC[:len(g[:10])]).sum() / idcg]


def per_query(st, src):
    """-> {item_id: [R@1, R@3, R@5, NDCG@10]} over queries with a gold in the pool."""
    data = json.load(open(SETS[st]))
    kind, arg = src.split(":", 1)
    out = {}
    if kind == "dump":            # scripts/eval/score.py --out: per-query metrics exactly as score.py ranks
        D = json.load(open(f"{arg}/{st}.json"))
        return {i: r["m"] for i, r in D.items()}
    if kind == "scores":
        S = json.load(open(f"{arg}/{st}.json"))
        for it in data:
            gold = [bool(p["is_supporting"]) for p in it["paragraphs"]]; n_in = sum(gold)
            if n_in == 0:
                continue
            s = S[str(it["id"])]
            order = sorted(range(len(s)), key=lambda i: (-s[i], i))   # ties keep the first-stage order
            out[str(it["id"])] = _row([gold[i] for i in order], n_in)
    elif kind == "official":
        files = sorted(glob.glob(arg.format(st=st)), key=lambda f: int(re.search(r"\.s(\d+)\.json$", f).group(1)))
        n = len(files)
        for s_idx, f in enumerate(files):
            for j, r in enumerate(json.load(open(f))):
                it = data[s_idx + j * n]
                if r["n_in"] == 0:
                    continue
                h, n_in = r["hits"], r["n_in"]
                out[str(it["id"])] = [h["1"] / n_in, h["3"] / n_in, h["5"] / n_in, r["nd10"]]
    else:
        raise ValueError(src)
    return out


def paired(a, b, n_perm=100_000, n_boot=20_000, seed=0):
    """a, b: (N,) per-query values -> dict(mean_a, mean_b, delta, p_rand, p_t, ci_lo, ci_hi)."""
    from scipy import stats
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    hits = 0
    for lo in range(0, n_perm, 10_000):
        m = min(10_000, n_perm - lo)
        signs = rng.integers(0, 2, size=(m, d.size), dtype=np.int8) * 2 - 1
        hits += int((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum())
    p_rand = (hits + 1) / (n_perm + 1)
    p_t = float(stats.ttest_rel(a, b).pvalue) if d.any() else 1.0
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(1)
    return dict(mean_a=100 * np.mean(a), mean_b=100 * np.mean(b), delta=100 * d.mean(), p_rand=p_rand,
                p_t=p_t, ci_lo=100 * np.percentile(boots, 2.5), ci_hi=100 * np.percentile(boots, 97.5), n=d.size)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def compare(st, src_a, src_b):
    A, B = per_query(st, src_a), per_query(st, src_b)
    assert set(A) == set(B), f"{st}: query sets differ ({len(A)} vs {len(B)})"
    ids = sorted(A)
    res = {}
    for mi, m in enumerate(METRICS):
        res[m] = paired([A[i][mi] for i in ids], [B[i][mi] for i in ids])
    return res


if __name__ == "__main__":
    # usage: python scripts/baselines/sig_test.py <set> <srcA> <srcB>
    if len(sys.argv) != 4 or sys.argv[1] not in SETS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    st, sa, sb = sys.argv[1:4]
    for m, r in compare(st, sa, sb).items():
        print(f"{st:9s} {m:8s} A {r['mean_a']:6.2f}  B {r['mean_b']:6.2f}  delta {r['delta']:+6.2f} "
              f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  p_rand {r['p_rand']:.2g}  p_t {r['p_t']:.2g}  {stars(r['p_rand'])}  (n={r['n']})")
