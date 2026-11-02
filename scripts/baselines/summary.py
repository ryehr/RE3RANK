"""Macro-9 table of conditional R@3 / NDCG@10 over any set of result directories (baselines and RE3Rank
runs mix freely).

Each argument is DIR or DIR=LABEL, where DIR holds one metrics file <set>.txt per evaluation set, as
written by run_baseline.sh / run_qrranker.sh ($RE3RANK_OUT/baselines/<tag>) or by scripts/eval/run_eval.sh
($RE3RANK_OUT/eval/<tag>/scores). A bare name without a slash is read as $RE3RANK_OUT/baselines/<name>.

  python scripts/baselines/summary.py first-stage=First-stage qrranker=QRRanker \\
      outputs/eval/re3rank-4b_final/scores=RE3Rank-4B
"""
import os
import re
import sys

from re3rank import paths

SETS = ["hotpotqa", "2wiki", "musique", "nq", "popqa", "narrativeqa", "detectiveqa", "locomo", "longmemeval"]
arms = [a.split("=", 1) if "=" in a else [a, a] for a in sys.argv[1:]]
if not arms or any(a in ("-h", "--help") for a, _ in arms):
    print(__doc__)
    sys.exit(0)


def result_dir(t):
    return t if "/" in t else paths.out("baselines", t)


print("| set | " + " | ".join(lbl for _, lbl in arms) + " |")
mac = {t: [[], []] for t, _ in arms}
for st in SETS:
    row = [st]
    for t, _ in arms:
        try:
            txt = open(os.path.join(result_dir(t), f"{st}.txt")).read()
            seg = txt[txt.index("conditional"):]
            r3 = float(re.search(r"R@3=([\d.]+)", seg).group(1))
            nd = float(re.search(r"NDCG@10=([\d.]+)", seg).group(1))
            mac[t][0].append(r3); mac[t][1].append(nd)
            row.append(f"{r3:.2f}/{nd:.2f}")
        except Exception:
            row.append("--")
    print("| " + " | ".join(row) + " |")
print("| **macro-9** | " + " | ".join(
    f"**{sum(mac[t][0])/len(mac[t][0]):.2f}/{sum(mac[t][1])/len(mac[t][1]):.2f}**"
    if len(mac[t][0]) == 9 else
    (f"{sum(mac[t][0])/len(mac[t][0]):.2f}/{sum(mac[t][1])/len(mac[t][1]):.2f} ({len(mac[t][0])}/9)"
     if mac[t][0] else "--")
    for t, _ in arms) + " |")
