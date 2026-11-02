"""Score one candidate file with a head scorer and report the conditional metrics.

With --traces the query rows are each item's reasoning trace (RE3Rank); without, they are the
question (the untrained QRHead-style baseline). Items with no trace in the file fall back to the
question.

  python scripts/eval/score.py --model Qwen/Qwen3-4B --heads 23-10,21-8 \
      --ckpt outputs/ckpt/re3rank-4b/final/scorer.pt \
      --traces outputs/eval/re3rank-4b/traces/musique.json --data musique

Prints the conditional line "R@1=.. R@3=.. R@5=.. R@10=.. NDCG@10=..". --out also writes, per item,
the 50 scores and [R@1, R@3, R@5, NDCG@10] (null for an item without a gold candidate, which is only
kept with --keep-nogold), e.g. for paired significance tests or for downstream QA.
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from re3rank import paths
from re3rank.metrics import KS, query_metrics, summary_line
from re3rank.scorer import build_instance, load_scorer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="backbone (scorer trunk and tokenizer), e.g. Qwen/Qwen3-4B")
    ap.add_argument("--heads", required=True, help="scoring heads 'L-H,L-H'")
    ap.add_argument("--ckpt", default="", help="trained scorer.pt (empty = untrained backbone)")
    ap.add_argument("--traces", default="", help="comma-separated {id: trace} JSON files (empty = question as query)")
    ap.add_argument("--data", required=True, help="candidate file, or the name of an evaluation set")
    ap.add_argument("--gold-denom", type=int, default=0, help=
                    "also report absolute recall with this many golds per query (2 for HotpotQA)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="", help="per-item scores and metrics (JSON)")
    ap.add_argument("--keep-nogold", action="store_true", help="--out also covers items without a gold candidate")
    ap.add_argument("--lowmem", action="store_true", help=
                    "store the trunk's Linear weights in bf16 (it runs under bf16 autocast anyway, so the "
                    "scores are identical); the scoring heads' own q/k projections stay fp32")
    args = ap.parse_args()

    traces = {}
    for f in filter(None, args.traces.split(",")):
        traces.update(json.load(open(f)))
    print(f"{len(traces)} traces" if traces else "no traces: query rows = the question")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    net = load_scorer(args.model, args.heads, args.ckpt)
    if args.lowmem:
        keep = {id(m) for sh in net.score_heads.values() for m in (sh.q_proj, sh.k_proj)}
        for m in net.trunk.modules():
            if isinstance(m, torch.nn.Linear) and id(m) not in keep:
                m.to(torch.bfloat16)
    net = net.to("cuda").eval()

    path = paths.eval_file(args.data) if args.data in paths.EVAL_SETS else args.data
    data = json.load(open(path))
    if args.limit:
        data = data[: args.limit]

    cond = {k: [] for k in KS}
    absr = {k: [] for k in KS}
    nd10, dump = [], {}
    for qi, item in enumerate(data):
        inst = build_instance(item, tokenizer, args.model, traces.get(str(item["id"])))
        n_in = int(inst["gold"].sum().item())
        denom_abs = args.gold_denom or item.get("n_gold", 0)
        if n_in == 0:
            for k in KS:
                if denom_abs:
                    absr[k].append(0.0)
            if args.keep_nogold:
                scores, _ = net(inst["input_ids"].to("cuda"), inst["q_lo"], inst["q_hi"], inst["chunk_ranges"])
                dump[str(item["id"])] = {"s": scores.float().cpu().tolist(), "m": None}
            continue
        scores, _ = net(inst["input_ids"].to("cuda"), inst["q_lo"], inst["q_hi"], inst["chunk_ranges"])
        rec, nd = query_metrics(scores, inst["gold"])
        nd10.append(nd)
        for k in KS:
            cond[k].append(rec[k])
            if denom_abs:
                absr[k].append(round(rec[k] * n_in) / denom_abs)   # hits / denom
        dump[str(item["id"])] = {"s": scores.float().cpu().tolist(), "m": [rec[1], rec[3], rec[5], nd]}
        if (qi + 1) % 200 == 0:
            print(f"[{qi+1}/{len(data)}] R@3(cond)={100*sum(cond[3])/len(cond[3]):.2f}", flush=True)

    print("\n=== conditional (denom = golds in candidates) ===")
    print(summary_line(cond, nd10) + f"  (n={len(nd10)})")
    if any(absr[3]):
        print("=== absolute ===")
        print("  " + "  ".join(f"R@{k}={100*sum(v)/len(v):.2f}" for k, v in absr.items()))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dump, open(args.out, "w"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
