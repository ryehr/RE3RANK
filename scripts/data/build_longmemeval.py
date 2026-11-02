"""LongMemEval-S candidates from the HF dataset xiaowu0162/longmemeval (file longmemeval_s).

Questions: the 500 of LongMemEval-S minus the 30 abstention questions (question_id ending in
"_abs"), 470. The retrieval pool of a question is its own haystack (about 50 sessions): every
session is cut into windows of 2 consecutive messages, each message written "role: content" with
its content capped at 200 Qwen3 tokens; a session's last single message is merged into the
previous window, and every window starts with "[<session date>] ". Gold: the windows of the
evidence sessions (answer_session_ids) that hold a message flagged has_answer; an evidence session
without any flag counts all its windows as gold. Candidates: the cosine top-50 windows
(Qwen3-Embedding-8B over the window text); a question whose gold windows all miss the top-50 is
dropped (466 remain). A candidate's title is its window id "<haystack session id>-w<k>".

Stages, under <data>/work/longmemeval/:
  python scripts/data/build_longmemeval.py parse                         # CPU
  python scripts/data/build_longmemeval.py embed --shard i --nshards 4   # GPU, one per shard
  python scripts/data/build_longmemeval.py assemble                      # -> paths.EVAL_SETS["longmemeval"]
assemble concatenates the shards in shard order, so the item order of the released file is that of
--nshards 4.
"""
import argparse
import collections
import glob
import os

from re3rank import paths

from _data_common import TOPK, hf_file, load_embedder, load_json, load_tokenizer, save_json, work

HF_REPO = "xiaowu0162/longmemeval"
HF_REVISION = "2ec2a557f339b6c0369619b1ed5793734cc87533"
HF_FILE = "longmemeval_s"
WIN, MSG_CAP = 2, 200   # messages per window, Qwen3 tokens per message


def wpath(name):
    return work("longmemeval", name)


def windows_of(turns, tok):
    """[(window text, any has_answer flag)] of one session."""
    lines, flags = [], []
    for t in turns:
        txt = t["content"]
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        if len(ids) > MSG_CAP:
            txt = tok.decode(ids[:MSG_CAP])
        lines.append(f"{t['role']}: {txt}")
        flags.append(bool(t.get("has_answer")))
    out = []
    for s in range(0, len(lines), WIN):
        e = min(s + WIN, len(lines))
        if len(lines) - s < 2 and out:
            pt, pf = out[-1]
            out[-1] = (pt + "\n" + "\n".join(lines[s:e]), pf or any(flags[s:e]))
            continue
        out.append(("\n".join(lines[s:e]), any(flags[s:e])))
    return out


def parse_item(x, tok):
    """(windows, gold window indices, evidence sessions without any has_answer flag)"""
    ev_ids = set(x["answer_session_ids"])
    wins, gold, n_fallback = [], [], 0
    for sid, sess, date in zip(x["haystack_session_ids"], x["haystack_sessions"],
                               x["haystack_dates"]):
        is_ev = sid in ev_ids
        ws = windows_of(sess, tok)
        flagged = any(f for _, f in ws)
        for k, (txt, f) in enumerate(ws):
            g = (f if flagged else True) if is_ev else False
            if g:
                gold.append(len(wins))
            wins.append({"cid": f"{sid}-w{k}", "date": date, "text": f"[{date}] {txt}"})
        if is_ev and not flagged:
            n_fallback += 1
    return wins, gold, n_fallback


def stage_parse(args):
    tok = load_tokenizer()
    data = load_json(hf_file(HF_REPO, HF_FILE, HF_REVISION))
    items = []
    n_fallback = 0
    for x in data:
        if str(x["question_id"]).endswith("_abs") or not x["answer_session_ids"]:
            continue
        wins, gold, nf = parse_item(x, tok)
        n_fallback += nf
        if not gold:
            continue
        items.append({"qid": x["question_id"], "qtype": x["question_type"],
                      "question": x["question"], "gold": gold, "windows": wins})
    save_json(items, wpath("parsed.json"))
    nw = sum(len(x["windows"]) for x in items)
    ng = [len(x["gold"]) for x in items]
    print(f"parsed {len(items)} items, {nw:,} windows, gold windows/item "
          f"mean {sum(ng)/len(ng):.2f}; session-fallback golds: {n_fallback}")


def make_item(x, top, gset):
    wins = x["windows"]
    return {
        "id": f"lme-{x['qid']}", "qtype": x["qtype"],
        "question": x["question"], "n_gold": len(gset),
        "gold_titles": [wins[g]["cid"] for g in sorted(gset)],
        "paragraphs": [{
            "idx": int(ci), "title": wins[int(ci)]["cid"],
            "paragraph_text": wins[int(ci)]["text"],
            "is_supporting": int(ci) in gset,
        } for ci in top],
    }


def stage_embed(args):
    import numpy as np
    items = load_json(wpath("parsed.json"))[args.shard::args.nshards]
    model = load_embedder(max_seq_length=1024)
    out, n_gold, n_in = [], 0, 0
    for n, x in enumerate(items):
        wins = x["windows"]
        cemb = model.encode([w["text"] for w in wins], batch_size=48,
                            normalize_embeddings=True, convert_to_numpy=True
                            ).astype(np.float32)
        qemb = model.encode([x["question"]], prompt_name="query",
                            normalize_embeddings=True, convert_to_numpy=True
                            ).astype(np.float32)
        sims = (qemb @ cemb.T)[0]
        top = np.argsort(-sims)[:TOPK]
        gset = set(x["gold"])
        hits = sum(int(ci) in gset for ci in top)
        n_gold += len(gset)
        n_in += hits
        if hits == 0:
            continue
        out.append(make_item(x, top, gset))
        if (n + 1) % 25 == 0:
            print(f"shard {args.shard}: {n+1}/{len(items)}", flush=True)
    save_json(out, wpath(f"top50.s{args.shard}.json"))
    print(f"shard {args.shard} DONE: {len(out)}/{len(items)} kept | "
          f"gold-in-pool R@50 = {100*n_in/max(1, n_gold):.2f}", flush=True)


def stage_assemble(args):
    items = []
    for f in sorted(glob.glob(wpath("top50.s*.json"))):
        items += load_json(f)
    path = paths.eval_file("longmemeval")
    save_json(items, path)
    ty = collections.Counter(x["qtype"] for x in items)
    gi = [sum(p["is_supporting"] for p in x["paragraphs"]) for x in items]
    multi = sum(1 for g in gi if g >= 2)
    print(f"{len(items)} items | types {dict(ty)}")
    print(f"gold-in-pool/item mean {sum(gi)/len(gi):.2f}; items with >=2 gold "
          f"windows in pool: {multi} ({100*multi/len(items):.0f}%)")
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("parse", "embed", "assemble"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(wpath(""), exist_ok=True)
    {"parse": stage_parse, "embed": stage_embed, "assemble": stage_assemble}[args.stage](args)


if __name__ == "__main__":
    main()
