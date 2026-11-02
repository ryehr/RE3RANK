"""Joint training of the RE3Rank generator and scorer (one epoch over the training mixture).

Per item, the generator G (the backbone in thinking mode, plain system prompt, thinking block forced
open) samples a reasoning trace over the question and its 50 candidates. The scorer S (a second copy
of the backbone, truncated above its scoring heads) reads the prompt followed by the trace, and the
attention its heads pay from the trace tokens to each candidate gives the candidate scores. The
ranking loss on these scores (CE + 2 * Recall@k surrogate) trains S directly and G through a
straight-through estimator:

    the scorer's input embedding of trace token t is  E[u_t] + p_t @ E - (p_t @ E).detach()

where u_t is the sampled token and p_t = softmax(logits_t / tau) is G's distribution at that step,
recomputed by a teacher-forced pass over the sampled trace. The forward value is exactly the
embedding of the sampled token; the backward pass sends the loss gradient into G's logits. The trace
tokens enter S by direct token splicing (no decode / re-tokenise), so trace positions align 1:1.

Layout: one GPU per rank holds G (bf16) and S (fp32); --two-gpu puts S on a second GPU per rank
(needed for 8B). Both are wrapped in DDP; the input embedding table of S stays frozen.

  torchrun --nproc_per_node 8 scripts/train/train.py --base Qwen/Qwen3-4B --heads 23-10,21-8 \
      --out outputs/ckpt/re3rank-4b

Checkpoints: <out>/upd{N} every --save-every updates and <out>/final, each with the generator
weights, the tokenizer and scorer.pt; resume with --init <out>/upd{N} --upd-offset N.
"""
import argparse
import contextlib
import json
import math
import os
import random
import time as _time

import torch
import torch.nn as nn

from re3rank import backbone as bbk
from re3rank import paths
from re3rank.prompts import RR_SYSTEM, build_prompt, user_msg
from re3rank.scorer import HeadScorer, build_instance, parse_heads, ranking_loss

# flex_attention sampling is torch.compiled per shape class; the default recompile budget (8) is
# exhausted by the probe's odd last batch, after which the compiled kernel silently degrades
torch._dynamo.config.cache_size_limit = 64

MAX_PROMPT = 13000   # generator prompts longer than this (tokens) are skipped
NDCG_K = 10


def flat_ids(p_ids):
    if not isinstance(p_ids, list):
        p_ids = p_ids["input_ids"]
    if p_ids and isinstance(p_ids[0], list):
        p_ids = p_ids[0]
    return p_ids


def ndcg_from_scores(scores, gold_mask, k=NDCG_K):
    """Binary NDCG@k, ideal DCG over min(k, |gold|)."""
    order = torch.argsort(scores.float(), descending=True).tolist()
    gold = set(i for i, g in enumerate(gold_mask.tolist()) if g)
    if not gold:
        return 0.0
    disc = [1.0 / math.log2(j + 2) for j in range(k)]
    dcg = sum(disc[j] for j, i in enumerate(order[:k]) if i in gold)
    return dcg / sum(disc[: min(k, len(gold))])


_POOL = {}


def _prompt_ids_pool(x):
    """Generator prompt ids of one item (fork-pool worker): plain system prompt, native thinking
    template, thinking block forced open, so the sampled tokens are exactly the trace S reads."""
    tok = _POOL["tok"]
    ids = flat_ids(tok.apply_chat_template(
        [{"role": "system", "content": RR_SYSTEM},
         {"role": "user", "content": user_msg(x["question"], x["paragraphs"])}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True))
    return ids + list(_POOL["think_open"])


def scorer_pieces(item, tok, model_id):
    """Prefix ids of the scorer input (up to and including the thinking opener), the token range of
    every candidate, and the gold mask. The trace ids are appended after the prefix."""
    f = bbk.fmt_for(model_id)
    full_seq, chunk_ranges = build_prompt(item["question"], item["paragraphs"], f.user_open)
    pre = f.sys_pre(RR_SYSTEM)
    prefix_str = pre + full_seq + f.assistant_open + f.think_open
    enc = tok(prefix_str, return_tensors=None, return_offsets_mapping=True, add_special_tokens=False)
    char_to_token = {}
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        for j in range(s, e):
            char_to_token[j] = i
    tcr = [[char_to_token.get(a + len(pre), 0), char_to_token.get(b + len(pre) - 1, 0) + 1]
           for a, b in chunk_ranges]
    gold = torch.tensor([bool(p["is_supporting"]) for p in item["paragraphs"]])
    return flat_ids(enc), tcr, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B", help="backbone of both models (Qwen3 0.6B / 4B / 8B)")
    ap.add_argument("--heads", required=True, help="scoring heads 'L-H,L-H' (see scripts/heads/)")
    ap.add_argument("--data", default=paths.data(paths.TRAIN_MIX), help="training candidate file")
    ap.add_argument("--out", required=True, help="checkpoint directory")
    ap.add_argument("--init", default="", help="generator init: the base model, or a saved upd dir to resume "
                    "(its scorer.pt is loaded too)")
    ap.add_argument("--upd-offset", type=int, default=0, help="update number to continue from on resume")
    ap.add_argument("--updates", type=int, default=0, help="updates to run (0 = to the end of the epoch)")
    ap.add_argument("--eps-per-rank", type=int, default=12, help="items per rank per update")
    ap.add_argument("--ep-batch", type=int, default=4, help="items per teacher-forced / scorer micro-batch")
    ap.add_argument("--gen-batch", type=int, default=12, help="prompts per sampling call")
    ap.add_argument("--resp-len", type=int, default=1024, help="max sampled trace tokens")
    ap.add_argument("--sample-temp", type=float, default=1.0, help="rollout temperature (<= 0: greedy)")
    ap.add_argument("--tau", type=float, default=1.0, help="softmax temperature of the straight-through path")
    ap.add_argument("--lr", type=float, default=1e-5, help="generator learning rate")
    ap.add_argument("--scorer-lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--loss", default="ce+rsatk", choices=["ce", "rsatk", "ce+rsatk"])
    ap.add_argument("--rsk-mix", type=float, default=2.0, help="ce+rsatk: weight of the rsatk term")
    ap.add_argument("--rsk-ks", default="3,10", help="k values averaged in the rsatk term")
    ap.add_argument("--rsk-tau1", type=float, default=1.0, help="rsatk rank-axis temperature")
    ap.add_argument("--rsk-tau2", type=float, default=1.0, help="rsatk score-axis temperature")
    ap.add_argument("--gen-attn", default="flex_attention", help=
                    "attention implementation used only while sampling (left-padded batches of long "
                    "prompts); training passes switch back to the model's own implementation")
    ap.add_argument("--two-gpu", action="store_true", help="two GPUs per rank: G on 2i, S on 2i+1")
    ap.add_argument("--probe-every", type=int, default=50, help="greedy-decode NDCG@10 on fixed items every N updates")
    ap.add_argument("--probe-n", type=int, default=120)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()
    GEN_BATCH = args.gen_batch
    BASE_ID = args.base
    init = args.init or BASE_ID
    rsk_ks = tuple(int(x) for x in args.rsk_ks.split(","))

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    g0, g1 = (2 * local, 2 * local + 1) if args.two_gpu else (local, local)
    torch.cuda.set_device(g0)
    dev0 = torch.device(f"cuda:{g0}")
    dev1 = torch.device(f"cuda:{g1}")

    def score_loss(scores, gold):
        return ranking_loss(scores, gold, args.loss, rsk_ks, args.rsk_tau1, args.rsk_tau2, args.rsk_mix)

    FMT = bbk.fmt_for(BASE_ID)      # chat / reasoning-channel literals of the backbone
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    eos = tok.eos_token_id
    # forced-open reasoning marker (Qwen '<think>' = 1 id; Gemma-4 '<|channel>thought\n' = 3 ids)
    # and the closing id used as a stop
    think_open = FMT.think_open_ids(tok)
    stop_ids = FMT.stop_ids(tok)

    # ---- data: items with at least one gold and one negative candidate, prompts <= MAX_PROMPT
    src = json.load(open(args.data))
    pool = [x for x in src if 0 < sum(p["is_supporting"] for p in x["paragraphs"]) < len(x["paragraphs"])]
    # every rank tokenizes the whole pool: fork pool sized to this rank's share of the cores
    _POOL.update(tok=tok, think_open=think_open)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 8) // world)) as _pl:
        all_ids = _pl.map(_prompt_ids_pool, pool, chunksize=32)
    eps = [(ids, x) for ids, x in zip(all_ids, pool) if len(ids) <= MAX_PROMPT]
    probe = random.Random(7).sample(eps, min(args.probe_n, len(eps)))
    # one epoch in a fixed global shuffle, rank-strided: each item exactly once, and resuming with
    # --upd-offset continues the same order
    perm = list(range(len(eps)))
    random.Random(1234).shuffle(perm)
    walk = perm[rank::world]
    n_epoch = -(-max(len(perm[r::world]) for r in range(world)) // args.eps_per_rank)
    n_updates = args.updates or (n_epoch - args.upd_offset)
    if rank == 0:
        print(f"pool: {len(eps)} prompts (of {len(pool)}) | probe {len(probe)} | epoch = {n_epoch} updates"
              f" | running {n_updates} from {args.upd_offset}", flush=True)

    # ---- generator G on g0. DDP needs the graph to start inside the wrapped forward, so wrap a
    # response-logits forward around the LM.
    class GenWrap(nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.lm = lm

        def forward(self, input_ids, ctx, Ts):
            h = self.lm.model(input_ids=input_ids).last_hidden_state
            # right-padded group: row i's response logits live at positions [ctx_i - 1, ctx_i - 1 + T_i)
            return [self.lm.lm_head(h[i, c - 1:c - 1 + t]) for i, (c, t) in enumerate(zip(ctx, Ts))]

    lm = AutoModelForCausalLM.from_pretrained(init, torch_dtype=torch.bfloat16)
    bbk.freeze_non_text(lm)   # Gemma-4 wrapper: vision/audio towers stay frozen
    bbk.freeze_ple(lm)        # Gemma-4 E-models: per-layer-embedding table
    lm.config.use_cache = False
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gw = GenWrap(lm).to(dev0)
    gw.train()   # from_pretrained returns eval mode, which silently disables gradient checkpointing
    TRAIN_ATTN = lm.config._attn_implementation

    def gen_mode(m):
        """sampling: checkpointing off, KV cache on, fast masked-decode attention"""
        m.gradient_checkpointing_disable()
        m.config.use_cache = True
        if args.gen_attn:
            try:
                m.set_attn_implementation(args.gen_attn)
            except Exception as e:   # backbone without that kernel: keep the training implementation
                if rank == 0 and not getattr(gen_mode, "_warned", False):
                    print(f"[gen] --gen-attn {args.gen_attn} unavailable ({e}); using {TRAIN_ATTN}", flush=True)
                    gen_mode._warned = True

    def train_mode(m):
        m.config.use_cache = False
        if args.gen_attn:
            m.set_attn_implementation(TRAIN_ATTN)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.train()

    ddp = DDP(gw, device_ids=[g0], gradient_as_bucket_view=True)

    # ---- scorer S on g1 (fp32 weights, bf16 autocast inside), trained jointly
    net = HeadScorer(BASE_ID, parse_heads(args.heads)).to(dev1, dtype=torch.float32)
    net.trunk.config.use_cache = False
    net.trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    net.train()
    for p in net.parameters():
        p.requires_grad_(True)
    # the inputs_embeds path never calls embed_tokens inside the DDP forward -> keep it frozen (it
    # must stay a constant for the straight-through product anyway)
    net.trunk.embed_tokens.weight.requires_grad_(False)
    if getattr(net.trunk, "embed_tokens_per_layer", None) is not None:
        net.trunk.embed_tokens_per_layer.weight.requires_grad_(False)   # Gemma PLE table
    sc_ck = os.path.join(init, "scorer.pt")
    if os.path.isfile(sc_ck):
        net.load_state_dict(torch.load(sc_ck, map_location=dev1))
        if rank == 0:
            print(f"[scorer] resumed from {sc_ck}", flush=True)
    E = net.trunk.embed_tokens.weight  # (V, hid) fp32, frozen
    if net.embed_scale != 1.0:         # Gemma: embed_tokens multiplies by sqrt(hidden)
        E = (E.detach() * net.embed_scale).contiguous()
    ddp_sc = DDP(net, device_ids=[g1], gradient_as_bucket_view=True)
    net = ddp_sc.module

    try:
        import bitsandbytes as bnb
        mk_opt = lambda ps, lr: bnb.optim.PagedAdamW8bit(ps, lr=lr)
        opt_name = "PagedAdamW8bit"
    except ImportError:
        mk_opt = lambda ps, lr: torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
        opt_name = "torch AdamW"
    opt = mk_opt(ddp.parameters(), args.lr)
    opt_sc = mk_opt([p for p in net.parameters() if p.requires_grad], args.scorer_lr)
    if rank == 0:
        print(f"[setup] base {BASE_ID} | heads {args.heads} | init {init} | loss {args.loss} | "
              f"optimiser {opt_name}", flush=True)

    def probe_phi(upd):
        """greedy traces on the fixed probe items, scored by the current S (NDCG@10)"""
        net.eval()
        m = ddp.module.lm
        gen_mode(m)
        share = probe[rank::world]
        vals, dumps = [], []
        with torch.no_grad():
            for i in range(0, len(share), GEN_BATCH):
                chunk = share[i:i + GEN_BATCH]
                n_real = len(chunk)
                # keep the sampling batch shape: pad the last chunk with repeats so the compiled
                # flex kernel is not recompiled for a new batch size; extra rows are discarded
                chunk = chunk + [chunk[0]] * (GEN_BATCH - n_real)
                maxlen = max(len(c[0]) for c in chunk)
                pad = torch.full((len(chunk), maxlen), eos, dtype=torch.long)
                msk = torch.zeros((len(chunk), maxlen), dtype=torch.long)
                for j, (ids, _) in enumerate(chunk):
                    pad[j, maxlen - len(ids):] = torch.tensor(ids)
                    msk[j, maxlen - len(ids):] = 1
                chunk = chunk[:n_real]
                out = m.generate(input_ids=pad.to(dev0), attention_mask=msk.to(dev0),
                                 max_new_tokens=args.resp_len, do_sample=False,
                                 pad_token_id=eos, eos_token_id=stop_ids)
                for j, (ids, item) in enumerate(chunk):
                    ev = tok.decode(out[j, maxlen:].tolist(), skip_special_tokens=True)
                    ev = ev if ev.strip() else ""
                    gold = [gi + 1 for gi, p in enumerate(item["paragraphs"]) if p["is_supporting"]]
                    v = 0.0
                    if ev and gold:
                        try:
                            inst = build_instance(item, tok, BASE_ID, trace=ev)
                            sc, _ = net(inst["input_ids"].to(dev1), inst["q_lo"], inst["q_hi"],
                                        inst["chunk_ranges"])
                            v = ndcg_from_scores(sc.cpu(), inst["gold"])
                        except (ValueError, IndexError):
                            pass
                    vals.append(v)
                    if len(dumps) < 2:
                        dumps.append(ev[:220].replace("\n", " "))
        train_mode(m)
        net.train()
        t = torch.tensor([sum(vals), len(vals)], device=dev0)
        dist.all_reduce(t)
        if rank == 0:
            print(f"[probe {upd}] greedy NDCG@10 {t[0].item()/max(t[1].item(), 1):.4f}"
                  f" (n={int(t[1].item())})", flush=True)
            for d in dumps:
                print(f"[probe {upd}] sample: {d}", flush=True)

    def save(d):
        ddp.module.lm.save_pretrained(d)
        tok.save_pretrained(d)
        bbk.copy_aux_files(BASE_ID, d)
        torch.save(net.state_dict(), f"{d}/scorer.pt")
        open(f"{d}/SAVED", "w").close()   # written last: marks a complete checkpoint

    if args.upd_offset == 0:
        probe_phi(0)
    for upd in range(args.upd_offset + 1, args.upd_offset + n_updates + 1):
        lo = ((upd - 1) * args.eps_per_rank) % len(walk)
        batch = [eps[i] for i in walk[lo:lo + args.eps_per_rank]]
        m = ddp.module.lm
        _tp0 = _time.time()

        # ---- rollout (no grad): sample one trace per item
        gen_mode(m)
        gen = []
        with torch.no_grad():
            for i in range(0, len(batch), GEN_BATCH):
                chunk = batch[i:i + GEN_BATCH]
                maxlen = max(len(c[0]) for c in chunk)
                pad = torch.full((len(chunk), maxlen), eos, dtype=torch.long)
                msk = torch.zeros((len(chunk), maxlen), dtype=torch.long)
                for j, (ids, _) in enumerate(chunk):
                    pad[j, maxlen - len(ids):] = torch.tensor(ids)
                    msk[j, maxlen - len(ids):] = 1
                samp = (dict(do_sample=True, temperature=args.sample_temp, top_p=1.0, top_k=0, min_p=0.0)
                        if args.sample_temp > 0 else dict(do_sample=False))
                out = m.generate(input_ids=pad.to(dev0), attention_mask=msk.to(dev0),
                                 max_new_tokens=args.resp_len, pad_token_id=eos,
                                 eos_token_id=stop_ids, **samp)
                for j, (ids, item) in enumerate(chunk):
                    resp = out[j, maxlen:].tolist()
                    cut = [k for k, t in enumerate(resp) if t in stop_ids]
                    if cut:
                        resp = resp[:cut[0] + 1]
                    T_ev = len(resp) - 1 if resp and resp[-1] in stop_ids else len(resp)
                    if T_ev > 0:
                        gen.append((ids, item, resp, T_ev))
        train_mode(m)
        torch.cuda.synchronize(dev0)
        _t_gen = _time.time() - _tp0
        _len_max = max((len(r) for _, _, r, _ in gen), default=0)
        _tp1 = _time.time()

        # equal item counts on every rank, or the DDP gradient sync diverges
        n_min = torch.tensor([len(gen)], device=dev0)
        dist.all_reduce(n_min, op=dist.ReduceOp.MIN)
        n_keep = int(n_min.item())
        if n_keep == 0:
            if rank == 0:
                print(f"[upd {upd}] a rank had no valid trace, skipped", flush=True)
            dist.barrier()
            continue
        gen = gen[:n_keep]

        opt.zero_grad(set_to_none=True)
        opt_sc.zero_grad(set_to_none=True)
        stat = {"loss": 0.0, "ndcg": 0.0, "len": 0.0, "n": 0}
        EB = args.ep_batch
        groups = [list(range(i, min(i + EB, n_keep))) for i in range(0, n_keep, EB)]
        for gi, gidx in enumerate(groups):
            grp = [gen[k] for k in gidx]
            sync = (gi == len(groups) - 1)      # all-reduce gradients once, on the last micro-batch
            cm = ddp.no_sync() if not sync else contextlib.nullcontext()
            cm2 = ddp_sc.no_sync() if not sync else contextlib.nullcontext()
            with cm, cm2:
                # teacher-forced pass of G over prompt + sampled trace (right-padded)
                fulls = [ids + resp for ids, _, resp, _ in grp]
                Lmax = max(len(f) for f in fulls)
                inp = torch.full((len(grp), Lmax), eos, dtype=torch.long)
                for i, f in enumerate(fulls):
                    inp[i, :len(f)] = torch.tensor(f)
                ctxs = [len(ids) for ids, _, _, _ in grp]
                Ts = [len(resp) for _, _, resp, _ in grp]
                lg_list = ddp(inp.to(dev0), ctxs, Ts)   # [(T_i, V)]
                embeds, spans, golds, lens, row_ids = [], [], [], [], []
                for i, (ids, item, resp, T_ev) in enumerate(grp):
                    logits = lg_list[i].float()
                    # straight-through trace embeddings on the scorer side
                    soft = torch.softmax(logits[:T_ev] / args.tau, dim=-1).to(dev1)
                    emb_soft = soft @ E                         # (T_ev, hid) fp32
                    ev_ids = torch.tensor(resp[:T_ev], device=dev1)
                    emb_ev = E[ev_ids].detach() + emb_soft - emb_soft.detach()
                    pre_ids, tcr, gold = scorer_pieces(item, tok, BASE_ID)
                    pe = E[torch.tensor(pre_ids, device=dev1)].detach()
                    embeds.append(torch.cat([pe, emb_ev], dim=0))
                    row_ids.append(list(pre_ids) + list(resp[:T_ev]))   # hard ids (Gemma PLE half only)
                    spans.append((len(pre_ids), len(pre_ids) + T_ev, tcr))
                    golds.append(gold)
                    lens.append(len(resp))
                Lm = max(e.shape[0] for e in embeds)
                eb = torch.zeros((len(grp), Lm, E.shape[1]), device=dev1, dtype=E.dtype)
                pid = torch.zeros((len(grp), Lm), device=dev1, dtype=torch.long)
                for i, (e, r) in enumerate(zip(embeds, row_ids)):
                    eb[i, :e.shape[0]] = e
                    pid[i, :len(r)] = torch.tensor(r, device=dev1)
                outs = ddp_sc(None, spans=spans, inputs_embeds=eb, ple_ids=pid)
                losses = [score_loss(sc, g.to(dev1)) for (sc, _), g in zip(outs, golds)]
                loss = sum(c.to(dev0) for c in losses) / n_keep
                loss.backward()
            for (sc, _), g, c, L in zip(outs, golds, losses, lens):
                stat["loss"] += c.item()
                stat["ndcg"] += ndcg_from_scores(sc.detach().cpu(), g)
                stat["len"] += L
                stat["n"] += 1
        if upd == args.upd_offset + 1 and rank == 0:
            print(f"[mem] g0 {torch.cuda.memory_allocated(dev0)/2**30:.1f} GiB"
                  f" | g1 {torch.cuda.memory_allocated(dev1)/2**30:.1f} GiB", flush=True)
        torch.cuda.synchronize(dev0)
        _t_fb = _time.time() - _tp1
        _tp2 = _time.time()
        gn = torch.nn.utils.clip_grad_norm_(ddp.parameters(), args.clip)
        gn_sc = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], args.clip)
        # DDP-averaged gradients are identical on every rank, so this decision is too
        if torch.isfinite(gn) and torch.isfinite(gn_sc):
            opt.step()
            opt_sc.step()
        elif rank == 0:
            print(f"[upd {upd}] non-finite grad norm (gen {gn.item():.3e} scorer {gn_sc.item():.3e})"
                  " -> update skipped", flush=True)
        torch.cuda.synchronize(dev0)
        _t_opt = _time.time() - _tp2
        tv = torch.tensor([_t_gen, _t_fb, _t_opt, float(_len_max)], device=dev0)
        dist.all_reduce(tv, op=dist.ReduceOp.MAX)
        agg = torch.tensor([stat["loss"], stat["ndcg"], stat["len"], stat["n"]], device=dev0)
        dist.all_reduce(agg)
        n = max(agg[3].item(), 1)
        if rank == 0:
            print(f"[upd {upd}] loss {agg[0].item()/n:.4f} | ndcg {agg[1].item()/n:.4f} | "
                  f"trace len {agg[2].item()/n:.1f} | mem peak {torch.cuda.max_memory_allocated(dev0)/2**30:.1f}"
                  f" GiB | gen {tv[0].item():.0f}s fb {tv[1].item():.0f}s opt {tv[2].item():.0f}s"
                  f" lmax {int(tv[3].item())}", flush=True)
        if args.probe_every and upd % args.probe_every == 0:
            probe_phi(upd)
        if args.save_every and upd % args.save_every == 0 and rank == 0:
            save(f"{args.out}/upd{upd}")
        dist.barrier()

    if rank == 0:
        save(f"{args.out}/final")
        print("TRAIN_DONE", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
