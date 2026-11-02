"""Scan all attention heads of a backbone for reasoning-retrieval heads.

On a seed sample of items, the base model (thinking mode, plain system prompt) writes its reasoning
trace; one forward pass over [system][user turn][assistant opener][<think>][trace] then measures,
for every full-attention head (l, h) and every candidate c,

  mass_q[l,h,c] = attention of the question tokens on c's tokens, averaged over the question tokens
                  (keys cut at the end of the question)                       -- QR, query = question
  mass_e[l,h,c] = the same for the trace tokens (keys cut at the end of the trace)
                                                                               -- RR, query = trace

and the report ranks the heads by gold attention mass, mean over items of sum_{c in gold} mass[l,h,c].

  stage 1 (vLLM, one shard per GPU):  python scripts/heads/scan_heads.py gen  --model Qwen/Qwen3-4B --out-dir D --shard i --nshards n
  stage 2 (HF,   one shard per GPU):  python scripts/heads/scan_heads.py scan --model Qwen/Qwen3-4B --out-dir D --shard i --nshards n
  stage 3 (CPU):                      python scripts/heads/scan_heads.py report --out-dir D
(scripts/heads/run_scan.sh runs all three.)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from re3rank import backbone as bbk
from re3rank import paths
from re3rank.attention import get_attn_weights
from re3rank.prompts import RR_SYSTEM, build_prompt, user_msg


def sample_items(sets, per_set, seed):
    """per_set items per set, drawn with Random(seed) from the items with a gold candidate"""
    rng = random.Random(seed)
    out = []
    for name in sets:
        data = json.load(open(paths.eval_file(name)))
        keep = [x for x in data if any(p.get("is_supporting") for p in x["paragraphs"])]
        for x in rng.sample(keep, min(per_set, len(keep))):
            out.append((name, x))
    return out


# ------------------------------------------------------------------ stage 1
def stage_gen(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    FMT = bbk.fmt_for(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    items = sample_items(args.sets.split(","), args.per_set, args.seed)[args.shard::args.nshards]
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": RR_SYSTEM},
         {"role": "user", "content": user_msg(x["question"], x["paragraphs"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for _, x in items]
    print(f"generating for {len(items)} items", flush=True)

    llm_kw = {"limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}} if FMT.fam == "gemma4" else {}
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, disable_log_stats=True, **llm_kw)
    outs = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=1024, skip_special_tokens=(FMT.fam == "qwen")))

    recs, bad = [], 0
    for (name, x), o in zip(items, outs):
        text = o.outputs[0].text
        # keep the reasoning RAW (no strip) so the scan layout is byte-identical to the generation
        if FMT.think_open in text:
            trace = text.split(FMT.think_open, 1)[1].split(FMT.think_close, 1)[0]
        else:
            trace = text          # truncated before the tag closed / no tag
        if len(trace.strip()) < 20:
            bad += 1
            continue
        recs.append({"set": name, "id": x["id"], "trace": trace})
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f"{args.out_dir}/traces.s{args.shard}.json"
    json.dump(recs, open(out_file, "w"), ensure_ascii=False)
    print(f"kept {len(recs)}, dropped {bad} (empty trace) -> {out_file}")


# ------------------------------------------------------------------ stage 2
@torch.no_grad()
def scan_one(model, tokenizer, item, trace, device):
    """-> (mass_q, mass_e), each (L, H, C); None if the spans cannot be located"""
    fam = bbk.family(model.config)
    FMT = bbk.Fmt(fam)
    body, ranges = build_prompt(item["question"], item["paragraphs"], FMT.user_open)
    pre = FMT.sys_pre(RR_SYSTEM)
    full = pre + body + f"{FMT.assistant_open}{FMT.think_open}{trace}"
    ranges = [[s + len(pre), e + len(pre)] for s, e in ranges]

    enc = tokenizer(full, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    c2t = {}
    for i, (s, e) in enumerate(enc["offset_mapping"][0].tolist()):
        for j in range(s, e):
            c2t[j] = i
    tcr = [[c2t.get(s, 0), c2t.get(e - 1, 0) + 1] for s, e in ranges]

    q = item["question"]
    q_start = full.index(q, len(pre))            # the one inside the user turn
    q_lo, q_hi = c2t[q_start], c2t[q_start + len(q) - 1] + 1
    ev_start = full.rindex(trace)
    inside = [i for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist())
              if a >= ev_start and b <= ev_start + len(trace) and b > a]
    if not inside:
        return None
    e_lo, e_hi = inside[0], inside[-1] + 1
    if not (q_hi > q_lo and e_hi > e_lo and e_lo > q_hi):
        return None

    captured, hooks = {}, []
    nl = len(model.layers)
    full_layers = [li for li in range(nl) if bbk.is_full(model, li)]   # sliding-window heads cannot see the chunks
    for li in full_layers:
        def mk(li):
            def hook(module, inp, out):
                captured[li] = (out[:, q_lo:q_hi].clone(), out[:, e_lo:e_hi].clone())
            return hook
        hooks.append(model.layers[li].self_attn.q_proj.register_forward_hook(mk(li)))
    try:
        out = model(input_ids=input_ids, use_cache=True)
    finally:
        for h in hooks:
            h.remove()

    pkv = out.past_key_values
    key_cache = (pkv.key_cache if hasattr(pkv, "key_cache") else [getattr(l, "keys", None) for l in pkv.layers])
    L = input_ids.shape[1]
    position_ids = torch.arange(L, device=device).unsqueeze(0)
    cos, sin = bbk.rope_cos_sin(model, torch.zeros(1, dtype=model.dtype, device=device),
                                position_ids, "full_attention")
    nh = model.config.num_attention_heads
    nc = len(item["paragraphs"])
    mass_q = torch.zeros(nl, nh, nc)
    mass_e = torch.zeros(nl, nh, nc)
    for li in full_layers:
        attn = model.layers[li].self_attn
        hd = attn.head_dim
        k_all = key_cache[bbk.kv_donor(model, li)]     # KV-shared layer -> its donor's cache
        for rows, lo, hi, upper, sink in ((captured[li][0], q_lo, q_hi, q_hi, mass_q),
                                          (captured[li][1], e_lo, e_hi, e_hi, mass_e)):
            qh = attn.q_norm(rows.view(1, hi - lo, nh, hd)).transpose(1, 2)
            qh = bbk.apply_rope(fam, qh, cos[:, lo:hi], sin[:, lo:hi])
            k = k_all[:, :, :upper]
            aw = get_attn_weights(k.float(), qh.float(), bbk.attn_scale(attn, fam)).mean(dim=-2)  # (1,H,upper)
            sink[li] = torch.stack([aw[0, :, s:e].sum(dim=-1) for s, e in tcr], dim=1).float().cpu()
    return mass_q.numpy(), mass_e.numpy()


def stage_scan(args):
    from transformers import AutoModel, AutoTokenizer
    recs = []
    for f in sorted(os.listdir(args.out_dir)):
        if f.startswith("traces.s") and f.endswith(".json"):
            recs += json.load(open(os.path.join(args.out_dir, f)))
    recs = recs[args.shard::args.nshards]
    by_id = {}
    for name in sorted({r["set"] for r in recs}):
        for x in json.load(open(paths.eval_file(name))):
            by_id[(name, x["id"])] = x

    device = "cuda"
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = bbk.text_model(model).to(device).eval()   # multimodal wrapper -> text decoder
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"scanning {len(recs)} items on {args.model}", flush=True)

    nl, nh, max_c, n = len(model.layers), model.config.num_attention_heads, 50, len(recs)
    MQ = np.zeros((n, nl, nh, max_c), dtype=np.float16)
    ME = np.zeros_like(MQ)
    GOLD = np.zeros((n, max_c), dtype=bool)
    SETS, k, t0 = [], 0, time.time()
    for r in recs:
        item = by_id[(r["set"], r["id"])]
        got = scan_one(model, tokenizer, item, r["trace"], device)
        if got is None:
            continue
        mq, me = got
        c = mq.shape[-1]
        MQ[k, :, :, :c] = mq.astype(np.float16)
        ME[k, :, :, :c] = me.astype(np.float16)
        GOLD[k, :c] = [bool(p.get("is_supporting")) for p in item["paragraphs"]]
        SETS.append(r["set"])
        k += 1
        if k % 25 == 0:
            print(f"  [{k}/{n}] {(time.time() - t0)/k:.2f}s/item", flush=True)
    np.savez_compressed(f"{args.out_dir}/scan.s{args.shard}.npz",
                        mass_q=MQ[:k], mass_e=ME[:k], gold=GOLD[:k], sets=np.array(SETS))
    print(f"saved {k} items -> {args.out_dir}/scan.s{args.shard}.npz")


# ------------------------------------------------------------------ stage 3
def load_scan(out_dir):
    mq, me, gold, sets = [], [], [], []
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("scan.s") and f.endswith(".npz"):
            d = np.load(os.path.join(out_dir, f))
            mq.append(d["mass_q"].astype(np.float32))
            me.append(d["mass_e"].astype(np.float32))
            gold.append(d["gold"])
            sets += list(d["sets"])
    return np.concatenate(mq), np.concatenate(me), np.concatenate(gold), np.array(sets)


def stage_report(args):
    mq, me, gold, sets = load_scan(args.out_dir)
    N, nl, nh, nc = mq.shape
    print(f"{N} items, {nl} layers x {nh} heads, gold/item {gold.sum(1).mean():.2f}\n")
    g = gold[:, None, None, :].astype(np.float32)
    gold_mass = lambda mass: (mass * g).sum(-1).mean(0)                        # (L, H)
    gold_share = lambda mass: ((mass * g).sum(-1) / (mass.sum(-1) + 1e-9)).mean(0)
    for tag, mass in (("QR (query = question)", mq), ("RR (query = reasoning trace)", me)):
        s, p = gold_mass(mass), gold_share(mass)
        idx = np.argsort(s.reshape(-1))[::-1][:12]
        print(f"### {tag}: top-12 heads by gold attention mass")
        print("| rank | head | gold mass | gold share |\n|---|---|---|---|")
        for r, i in enumerate(idx, 1):
            l, h = int(i // nh), int(i % nh)
            print(f"| {r} | {l}-{h} | {s[l, h]:.4f} | {p[l, h]:.3f} |")
        print()
    se = gold_mass(me).reshape(-1)
    top2 = [int(i) for i in np.argsort(se)[::-1][:2]]
    print("RR_TOP2 " + ",".join(f"{i // nh}-{i % nh}" for i in top2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("gen", "scan", "report"))
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sets", default="musique,narrativeqa", help="evaluation sets the seed sample is drawn from")
    ap.add_argument("--per-set", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=40960)
    args = ap.parse_args()
    {"gen": stage_gen, "scan": stage_scan, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
