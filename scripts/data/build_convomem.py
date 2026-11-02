"""ConvoMem training source (HF dataset Salesforce/ConvoMem, core_benchmark) and the second training
mixture.

Questions: 7,000 multi-evidence items -- changing 3,378 | implicit 1,622 | user 1,500 |
assistant 500. Only items with >= 2 evidence messages, each in its own conversation, are eligible;
abstention and preference (rubric-graded) items are not used. parse oversamples each category
(3,900 / 2,600 / 1,950 / 650, random.Random(3), round-robin over personas) and assemble keeps the
targets among the items that survive retrieval (random.Random(0)).

Pool, per persona: its filler conversations plus the evidence conversations of all its selected
items (so other items' evidence conversations are hard negatives), cut into windows of --win
messages; a message is "speaker: text" capped at --msg-cap Qwen3 tokens, and a tail of fewer than 3
messages joins the previous window. Gold: the window holding each evidence message, located by
exact text, then by the first 80 characters, then by whitespace/case-normalised equality.
Candidates: the cosine top-50 windows of the persona pool (Qwen3-Embedding-8B); an item whose gold
windows all miss the top-50 is dropped. A candidate's title is its window id: "ev<n>c<m>-w<k>" for
evidence conversations, "fill<j>-w<k>" for filler conversations.

Stages, under <data>/work/convomem/:
  python scripts/data/build_convomem.py parse                          # CPU
  python scripts/data/build_convomem.py split --nshards 4              # CPU, optional (less RAM per embed shard)
  python scripts/data/build_convomem.py embed --shard i --nshards 4    # GPU, one per shard (personas)
  python scripts/data/build_convomem.py assemble                       # CPU
assemble reads work/mix/mix_qaconv_friendsqa_squad.json, replaces its QAConv and FriendsQA items by
the 7,000 ConvoMem items and writes work/mix/mix_convomem_squad.json (38,000 items). It concatenates
the embed shards in shard order before sampling, so the released data used --nshards 4.
"""
import argparse
import collections
import glob
import os
import random
import re

from _data_common import (TOPK, hf_snapshot, load_embedder, load_json, load_tokenizer,
                          save_json, work)

HF_REPO = "Salesforce/ConvoMem"
HF_REVISION = "e3e9b39115b02346824c70d349350de738f8be41"
# category -> (final target, oversampled at parse time)
PLAN = {"changing": (3378, 3900), "implicit": (1622, 2600),
        "user": (1500, 1950), "assistant": (500, 650)}
CATDIR = {"changing": "changing_evidence", "implicit": "implicit_connection_evidence",
          "user": "user_evidence", "assistant": "assistant_facts_evidence"}


def wpath(name):
    return work("convomem", name)


def mix_in_path():
    return work("mix", "mix_qaconv_friendsqa_squad.json")


def mix_out_path():
    return work("mix", "mix_convomem_squad.json")


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def find_evidence(messages, ev_text):
    """Index of the message carrying ev_text, or None."""
    for i, m in enumerate(messages):
        if m["text"] == ev_text:
            return i
    key = ev_text[:80]
    for i, m in enumerate(messages):
        if key and key in m["text"]:
            return i
    nk = norm(ev_text)
    for i, m in enumerate(messages):
        if norm(m["text"]) == nk:
            return i
    return None


def windows_of(messages, tok, win, msg_cap):
    """[(text, lo, hi)] windows of `win` messages; a tail of < 3 messages is merged back."""
    lines = []
    for m in messages:
        t = m["text"]
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > msg_cap:
            t = tok.decode(ids[:msg_cap])
        lines.append(f"{m['speaker']}: {t}")
    spans = list(range(0, len(lines), win))
    out = []
    for s in spans:
        e = min(s + win, len(lines))
        if len(lines) - s < 3 and out:
            pt, pl, _ = out[-1]
            out[-1] = ("\n".join([pt] + lines[s:e]), pl, e)
            continue
        out.append(("\n".join(lines[s:e]), s, e))
    return out


def stage_parse(args):
    tok = load_tokenizer()
    patterns = [f"core_benchmark/evidence_questions/{d}/*" for d in CATDIR.values()]
    patterns.append("core_benchmark/filler_conversations/*")
    src = os.path.join(hf_snapshot(HF_REPO, HF_REVISION, allow_patterns=patterns),
                       "core_benchmark")
    rng = random.Random(3)

    # -------- select items, stratified over personas
    selected = collections.defaultdict(list)   # persona -> [(cat, item)]
    for cat, (_, take) in PLAN.items():
        cand = []
        for f in sorted(glob.glob(f"{src}/evidence_questions/{CATDIR[cat]}/*/*.json")):
            persona = os.path.basename(f).split("_")[0]
            for it in load_json(f)["evidence_items"]:
                if len(it.get("message_evidences", [])) >= 2 and \
                   len(it["conversations"]) == len(it["message_evidences"]):
                    cand.append((persona, it))
        rng.shuffle(cand)
        by_p = collections.defaultdict(list)
        for p, it in cand:
            by_p[p].append(it)
        # round-robin over personas so that no persona dominates
        order = sorted(by_p)
        got = 0
        while got < take and any(by_p[p] for p in order):
            for p in order:
                if got >= take:
                    break
                if by_p[p]:
                    selected[p].append((cat, by_p[p].pop()))
                    got += 1
        print(f"select {cat}: {got}/{take} (multi-evidence pool had {len(cand)})", flush=True)

    # -------- per persona: filler windows + evidence windows + questions
    personas_out = {}
    mstat = collections.Counter()
    fill_files = {os.path.basename(f).split("_")[0]: f
                  for f in glob.glob(f"{src}/filler_conversations/*.json")}
    for pi, (persona, items) in enumerate(sorted(selected.items())):
        wins, questions = [], []

        def add_conv(conv, tag):
            base = len(wins)
            ws = windows_of(conv["messages"], tok, args.win, args.msg_cap)
            for k, (t, lo, hi) in enumerate(ws):
                wins.append({"cid": f"{tag}-w{k}", "text": t, "lo": lo, "hi": hi})
            return base, ws
        if persona in fill_files:
            d = load_json(fill_files[persona])
            convs = [c for it in d["evidence_items"] for c in it["conversations"]]
            for j, conv in enumerate(convs):
                add_conv(conv, f"fill{j}")
        for n, (cat, it) in enumerate(items):
            gold = []
            for conv, mev in zip(it["conversations"], it["message_evidences"]):
                base, ws = add_conv(conv, f"ev{n}c{len(gold)}")
                idx = find_evidence(conv["messages"], mev["text"])
                if idx is None:
                    mstat["miss"] += 1
                    continue
                mstat["hit"] += 1
                for k, (t, lo, hi) in enumerate(ws):
                    if lo <= idx < hi:
                        gold.append(base + k)
                        break
            if gold:
                questions.append({"qid": f"cvm-{cat[:3]}-{persona[:8]}-{n}",
                                  "cat": cat, "question": it["question"],
                                  "gold": sorted(set(gold))})
            else:
                mstat["dropped_nomatch"] += 1
        personas_out[persona] = {"windows": wins, "questions": questions}
        if (pi + 1) % 20 == 0:
            print(f"parsed {pi+1}/{len(selected)} personas", flush=True)
    save_json(personas_out, wpath("personas.json"))
    nw = sum(len(v["windows"]) for v in personas_out.values())
    nq = sum(len(v["questions"]) for v in personas_out.values())
    print(f"parse done: {len(personas_out)} personas, {nw:,} windows, "
          f"{nq} questions | evidence match {dict(mstat)}", flush=True)


def stage_split(args):
    data = load_json(wpath("personas.json"))
    names = sorted(data)
    for k in range(args.nshards):
        sub = {n: data[n] for n in names[k::args.nshards]}
        save_json(sub, wpath(f"personas.s{k}of{args.nshards}.json"))
        print(f"shard {k}: {len(sub)} personas")


def make_item(q, wins, top):
    """The candidate item (with its category), or None when no gold is among the candidates."""
    gset = set(q["gold"])
    if not any(int(ci) in gset for ci in top):
        return None
    return {
        "id": q["qid"], "question": q["question"], "cat": q["cat"],
        "n_gold": len(gset),
        "gold_titles": [wins[g]["cid"] for g in sorted(gset)],
        "paragraphs": [{
            "idx": int(ci), "title": wins[int(ci)]["cid"],
            "paragraph_text": wins[int(ci)]["text"],
            "is_supporting": int(ci) in gset,
        } for ci in top],
    }


def stage_embed(args):
    import numpy as np
    import torch
    shard_file = wpath(f"personas.s{args.shard}of{args.nshards}.json")
    if os.path.exists(shard_file):        # pre-split by the split stage
        data = load_json(shard_file)
        names = sorted(data)
    else:
        data = load_json(wpath("personas.json"))
        names = sorted(data)[args.shard::args.nshards]
    model = load_embedder(max_seq_length=1024)
    out, n_gold_tot, n_in_tot = [], 0, 0
    for pn, persona in enumerate(names):
        v = data[persona]
        wins, qs = v["windows"], v["questions"]
        if not qs:
            continue
        cemb = model.encode([w["text"] for w in wins], batch_size=48,
                            normalize_embeddings=True, convert_to_numpy=True
                            ).astype(np.float32)
        qemb = model.encode([q["question"] for q in qs], prompt_name="query",
                            batch_size=64, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
        sims = torch.from_numpy(qemb).cuda() @ torch.from_numpy(cemb).cuda().T
        top_all = torch.argsort(sims, dim=1, descending=True)[:, :TOPK].cpu().numpy()
        for qi, q in enumerate(qs):
            top = top_all[qi]
            gset = set(q["gold"])
            n_gold_tot += len(gset)
            n_in_tot += sum(int(ci) in gset for ci in top)
            item = make_item(q, wins, top)
            if item is not None:
                out.append(item)
        if (pn + 1) % 5 == 0:
            print(f"shard {args.shard}: {pn+1}/{len(names)} personas, "
                  f"{len(out)} kept", flush=True)
    save_json(out, wpath(f"top50.s{args.shard}.json"))
    print(f"shard {args.shard} DONE: {len(out)} items | gold-in-pool "
          f"R@50 = {100*n_in_tot/max(1, n_gold_tot):.2f}", flush=True)


def stage_assemble(args):
    rng = random.Random(0)
    items = []
    for f in sorted(glob.glob(wpath("top50.s*.json"))):
        items += load_json(f)
    by_cat = collections.defaultdict(list)
    for x in items:
        by_cat[x["cat"]].append(x)
    final = []
    for cat, (target, _) in PLAN.items():
        pool = by_cat[cat]
        rng.shuffle(pool)
        take = pool[:target]
        print(f"{cat}: {len(take)} taken (of {len(pool)} surviving)")
        final += take
    for x in final:
        x.pop("cat", None)
    old = load_json(mix_in_path())
    keep = [x for x in old if not (str(x["id"]).startswith("qaconv")
                                   or str(x["id"]).startswith("friends"))]
    mixed = keep + final
    rng.shuffle(mixed)
    save_json(mixed, mix_out_path())
    print(f"{len(keep)} carried (MuSiQue / NarrativeQA / SQuAD) + {len(final)} ConvoMem "
          f"= {len(mixed)} -> {mix_out_path()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("parse", "split", "embed", "assemble"))
    ap.add_argument("--win", type=int, default=6, help="messages per window")
    ap.add_argument("--msg-cap", type=int, default=200, help="Qwen3 tokens per message")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(wpath(""), exist_ok=True)
    {"parse": stage_parse, "split": stage_split, "embed": stage_embed,
     "assemble": stage_assemble}[args.stage](args)
