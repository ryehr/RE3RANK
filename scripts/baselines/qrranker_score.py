"""Evaluate the released QRRanker checkpoint (MindscapeRAG/QRRanker: 16 trained QR heads on a
25-layer truncated Qwen3-4B-Instruct-2507) under the same protocol as the RE3Rank scorer without a
trace: the same top-50 candidate files, the same prompt construction (re3rank.prompts.build_prompt,
which drops the 'title: ' prefix when the title is empty -- this matters for NarrativeQA), the
question tokens as query rows, and the same conditional / absolute recall conventions.

The checkpoint's custom modeling code needs transformers 4.53 (docs/BASELINES.md): run this script with
that environment's python. get_cache_class and compute_qr_scores follow the code in the checkpoint's
model card (README of MindscapeRAG/QRRanker, Apache-2.0); compute_qr_scores takes the head list as
the [[layer, head], ...] list of the checkpoint's config.

  CUDA_VISIBLE_DEVICES=0 python scripts/baselines/qrranker_score.py --data musique \\
      --shard 0 --nshards 8 --out outputs/baselines/qrranker/musique.s0.json
  python scripts/baselines/qrranker_score.py --report outputs/baselines/qrranker/musique.s*.json

--out dumps one record per item ({n_in, denom, nd10, hits}, in data[shard::nshards] order) for --report
and for sig_test.py; --order-out also scores the items without a gold candidate and writes
{item id: [candidate positions, best first]} (downstream QA needs every item's ranking).
"""
import argparse
import importlib
import json
import math

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

from re3rank import paths
from re3rank.attention import get_attn_weights
from re3rank.prompts import build_prompt

MODEL_ID = "MindscapeRAG/QRRanker"


def get_cache_class(model):
    """Use the DynamicCacheWithQuery class from the module the model itself was
    loaded from -- an isinstance() check inside the model compares against that
    class object, so importing a separate copy silently disables query caching."""
    mod = importlib.import_module(type(model).__module__)
    return mod.DynamicCacheWithQuery


def compute_qr_scores(query_cache, key_cache, qr_head_list, chunk_ranges, query_upper_bound):
    all_head_scores = []
    for key_state, query_state in zip(key_cache, query_cache):
        attn_weights = get_attn_weights(key_state[:, :, :query_upper_bound, :], query_state)
        attn_weights = attn_weights.mean(dim=-2)
        chunk_scores = torch.stack(
            [attn_weights[:, :, s:e].sum(dim=-1) for s, e in chunk_ranges], dim=2
        )
        all_head_scores.append(chunk_scores)

    all_head_scores = torch.stack(all_head_scores, dim=1).float()
    indices = torch.tensor(qr_head_list, device=all_head_scores.device)
    all_head_scores = all_head_scores[:, indices[:, 0], indices[:, 1], :]
    return all_head_scores.sum(dim=1).squeeze(0)


@torch.no_grad()
def score_official(model, tokenizer, cache_cls, qr_head_list, item, device):
    full_seq, chunk_ranges = build_prompt(item["question"], item["paragraphs"])
    inputs = tokenizer(full_seq, max_length=262144, truncation=True,
                       return_tensors="pt", return_offsets_mapping=True,
                       add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    offset_mapping = inputs["offset_mapping"][0]

    char_to_token = {}
    for i, (s, e) in enumerate(offset_mapping.tolist()):
        for j in range(s, e):
            char_to_token[j] = i
    token_chunk_ranges = [
        [char_to_token.get(s, 0), char_to_token.get(e - 1, 0) + 1]
        for s, e in chunk_ranges
    ]
    q = item["question"]
    q_start = full_seq.rindex(q)
    query_positions = list(range(
        char_to_token[q_start], char_to_token[q_start + len(q) - 1] + 1))
    upper = query_positions[-1] + 1

    past_kv = cache_cls(query_indices=query_positions)
    output = model(input_ids, attention_mask, past_key_values=past_kv)
    scores = compute_qr_scores(
        output.past_key_values.query_cache, output.past_key_values.key_cache,
        qr_head_list, token_chunk_ranges, upper)
    return scores


KS = (1, 3, 5, 10)


def report(recs):
    cond = {k: [] for k in KS}
    absr = {k: [] for k in KS}
    for r in recs:
        for k in KS:
            if r["n_in"]:
                cond[k].append(r["hits"][str(k)] / r["n_in"])
            if r["denom"]:
                absr[k].append(r["hits"][str(k)] / r["denom"])
    nd = [r["nd10"] for r in recs if r["n_in"] and "nd10" in r]
    print(f"n={len(recs)} (scored {len(cond[3])})")
    print("=== conditional (denom = golds in candidates) ===")
    print("  " + "  ".join(f"R@{k}={100*sum(v)/len(v):.2f}" for k, v in cond.items())
          + (f"  NDCG@10={100*sum(nd)/len(nd):.2f}" if nd else ""))
    if any(absr[3]):
        print("=== absolute (paper convention) ===")
        print("  " + "  ".join(f"R@{k}={100*sum(v)/len(v):.2f}" for k, v in absr.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="candidate file, or the name of an evaluation set")
    ap.add_argument("--model", default=MODEL_ID, help="the QRRanker checkpoint (hub id or local dir)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gold-denom", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="", help="dump per-item records for --report")
    ap.add_argument("--order-out", default="", help="also score items with no gold in the pool and dump "
                    "{item id: [candidate positions, best first]} (downstream QA needs every item's ranking)")
    ap.add_argument("--report", nargs="+", default=[],
                    help="aggregate shard dumps and exit")
    args = ap.parse_args()

    if args.report:
        recs = [r for f in args.report for r in json.load(open(f))]
        report(recs)
        return

    device = "cuda"
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cache_cls = get_cache_class(model)

    data = json.load(open(paths.eval_file(args.data) if args.data in paths.EVAL_SETS else args.data))
    if args.limit:
        data = data[: args.limit]
    data = data[args.shard::args.nshards]

    recs, orders = [], {}
    for qi, item in enumerate(data):
        gold = torch.tensor([bool(p["is_supporting"]) for p in item["paragraphs"]])
        n_in = int(gold.sum().item())
        denom_abs = args.gold_denom or item.get("n_gold", 0)
        if n_in == 0 and not args.order_out:
            recs.append({"n_in": 0, "denom": denom_abs,
                         "hits": {str(k): 0 for k in KS}})
            continue
        scores = score_official(model, tokenizer, cache_cls, config.qr_head_list,
                                item, device)
        order = torch.argsort(scores, descending=True).cpu()
        if args.order_out:
            orders[str(item["id"])] = order.tolist()
        if n_in == 0:
            recs.append({"n_in": 0, "denom": denom_abs,
                         "hits": {str(k): 0 for k in KS}})
            continue
        dcg = sum(float(gold[order[i]]) / math.log2(i + 2)
                  for i in range(min(10, len(order))))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, n_in)))
        recs.append({"n_in": n_in, "denom": denom_abs, "nd10": dcg / idcg,
                     "hits": {str(k): int(gold[order[:k]].sum().item())
                              for k in KS}})
        if (qi + 1) % 100 == 0:
            done = [r for r in recs if r["n_in"]]
            r3 = 100 * sum(r["hits"]["3"] / r["n_in"] for r in done) / len(done)
            print(f"[{qi+1}/{len(data)}] R@3(cond)={r3:.2f}", flush=True)

    if args.out:
        json.dump(recs, open(args.out, "w"))
    if args.order_out:
        json.dump(orders, open(args.order_out, "w"))
    report(recs)


if __name__ == "__main__":
    main()
