"""Training sources QAConv, FriendsQA and SQuAD, and the first training mixture.

Human-written questions with their native gold; the pool of a source is its own chunk corpus,
shared by all its questions, and the candidates are the cosine top-50 over it (Qwen3-Embedding-8B).
Dialogue turns are written "speaker: text" and packed in order into chunks of at most --budget
Qwen3 tokens (a turn is never split; an overlong single turn, e.g. a long e-mail, is cut to the
budget).
  qaconv     QAConv V1.1 (github.com/salesforce/QAConv): all 18,728 article segments (train, val
             and test) -> 31,978 chunks. Questions: train only, human-written (no "QG" flag) and
             answerable; gold = the chunks of the question's own segment whose normalised text
             contains a reference answer (questions with no such chunk are dropped).
  friendsqa  FriendsQA (github.com/emorynlp/FriendsQA): all scenes of train, dev and test, packed per
             scene paragraph -> 2,397 chunks. Questions: train only; gold = the chunks holding an
             answer utterance.
  squad      SQuAD v1.1 train (HF rajpurkar/squad): the 18,891 unique paragraphs, title "Title"
             (underscores to spaces). Questions: deduplicated by normalised text, shuffled with
             random.Random(0), first 12,000; gold = the source paragraph.
A question is kept only if its top-50 holds at least one gold and at least one non-gold candidate.
The raw QAConv and FriendsQA files come from scripts/data/download_raw.sh.

  python scripts/data/build_qaconv_friendsqa_squad.py parse [--source S]     # CPU
  python scripts/data/build_qaconv_friendsqa_squad.py embed --source S       # GPU, one per source
  python scripts/data/build_qaconv_friendsqa_squad.py mix                    # CPU
mix draws MuSiQue-train 12,000 + NarrativeQA-train 12,000 + QAConv 5,000 + FriendsQA 2,000 +
SQuAD 7,000 = 38,000 items (random.Random(0)) into work/mix/mix_qaconv_friendsqa_squad.json; it
needs work/musique_train/candidates.json and work/narrativeqa/train_candidates.json. The final
training mixture keeps its QAConv and FriendsQA items (scripts/data/build_train_mix.py); SQuAD
contributes no item to it but has to be built, since its item count and positions fix the order of
the others.
"""
import argparse
import hashlib
import random
import re

from _data_common import (TOPK, load_embedder, load_json, load_tokenizer, raw, save_json,
                          work)

SQUAD_REPO = "rajpurkar/squad"
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
SOURCES = ("qaconv", "friends", "squad")
# work directory of each source (the item ids keep the short source names "friends-", ...)
SRC_DIR = {"qaconv": "qaconv", "friends": "friendsqa", "squad": "squad"}
TARGETS = {"qaconv": 5000, "friends": 2000, "squad": 7000,
           "musique": 12000, "nqa": 12000}


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def src_path(src, name):
    return work(SRC_DIR[src], name)


def mix_path():
    return work("mix", "mix_qaconv_friendsqa_squad.json")


def pack_turns(turns, tok, budget):
    """Pack turn strings into chunks <= budget tokens; returns
    (chunk_texts, turn_idx -> chunk_idx)."""
    texts, t2c = [], {}
    cur, cur_tok, cur_ids = [], 0, []

    def flush():
        nonlocal cur, cur_tok, cur_ids
        if cur:
            texts.append("\n".join(cur))
            for ti in cur_ids:
                t2c[ti] = len(texts) - 1
            cur, cur_tok, cur_ids = [], 0, []

    for ti, t in enumerate(turns):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:  # single overlong turn (e.g. e-mail bodies)
            t = tok.decode(ids[:budget])
            ids = ids[:budget]
        n = len(ids)
        if cur and cur_tok + n > budget:
            flush()
        cur.append(t)
        cur_ids.append(ti)
        cur_tok += n
    flush()
    return texts, t2c


# ---------------------------------------------------------------- parse
def parse_qaconv(tok, budget):
    seg = load_json(raw("qaconv", "QAConv-V1.1", "article_segment.json"))
    corpus, seg2chunks = [], {}
    for sid, s in seg.items():
        turns = [f"{t['speaker']}: {t['text']}" for t in s["seg_dialog"]]
        texts, _ = pack_turns(turns, tok, budget)
        seg2chunks[sid] = []
        for j, txt in enumerate(texts):
            seg2chunks[sid].append(len(corpus))
            corpus.append({"cid": f"{sid}#{j}", "text": txt})

    trn = load_json(raw("qaconv", "QAConv-V1.1", "trn.json"))
    qs, n_qg, n_unans, n_nomatch = [], 0, 0, 0
    for qa in trn:
        if qa.get("QG"):
            n_qg += 1
            continue
        answers = [a for a in qa.get("answers", []) if a]
        if not answers or any(norm(a) == "unanswerable" for a in answers):
            n_unans += 1
            continue
        gold = [ci for ci in seg2chunks.get(qa["article_segment_id"], [])
                if any(norm(a) in norm(corpus[ci]["text"]) for a in answers)]
        if not gold:
            n_nomatch += 1
            continue
        qs.append({"qid": qa["id"], "question": qa["question"], "gold": gold})
    print(f"qaconv: {len(corpus)} chunks from {len(seg)} segments; "
          f"{len(qs)} questions kept "
          f"(skipped {n_qg} QG, {n_unans} unanswerable, {n_nomatch} no-span-match)")
    return corpus, qs


def parse_friends(tok, budget):
    corpus, qs = [], []
    n_nomap = 0
    for split in ("trn", "dev", "tst"):
        data = load_json(raw("friendsqa", f"friendsqa_{split}.json"))["data"]
        for scene in data:
            title = scene["title"]
            for pi, para in enumerate(scene["paragraphs"]):
                utts = para.get("utterances:") or para.get("utterances")
                turns = [f"{', '.join(u['speakers'])}: {u['utterance']}"
                         for u in utts]
                texts, t2c = pack_turns(turns, tok, budget)
                base = len(corpus)
                for j, txt in enumerate(texts):
                    corpus.append({"cid": f"{title}#{pi}#{j}", "text": txt})
                if split != "trn":
                    continue
                uid2chunk = {utts[ti]["uid"]: base + ci for ti, ci in t2c.items()}
                for qa in para["qas"]:
                    gold = sorted({uid2chunk[a["utterance_id"]]
                                   for a in qa["answers"]
                                   if a["utterance_id"] in uid2chunk})
                    if not gold:
                        n_nomap += 1
                        continue
                    qs.append({"qid": qa["id"], "question": qa["question"],
                               "gold": gold})
    print(f"friends: {len(corpus)} chunks; {len(qs)} trn questions "
          f"({n_nomap} dropped, no utterance map)")
    return corpus, qs


def parse_squad():
    from datasets import load_dataset
    ds = load_dataset(SQUAD_REPO, split="train", revision=SQUAD_REVISION)
    corpus, ctx2idx, qs, seen_q = [], {}, [], set()
    for ex in ds:
        ctx = ex["context"]
        if ctx not in ctx2idx:
            h = hashlib.md5(ctx.encode()).hexdigest()[:8]
            ctx2idx[ctx] = len(corpus)
            corpus.append({"cid": f"{ex['title']}#{h}",
                           "title": ex["title"].replace("_", " "),
                           "text": ctx})
        nq = norm(ex["question"])
        if nq in seen_q:
            continue
        seen_q.add(nq)
        qs.append({"qid": ex["id"], "question": ex["question"],
                   "gold": [ctx2idx[ctx]]})
    rng = random.Random(0)
    rng.shuffle(qs)
    qs = qs[:12000]  # pre-embedding subsample; the mix takes its share of the survivors
    print(f"squad: {len(corpus)} unique paragraphs; {len(qs)} questions sampled")
    return corpus, qs


def stage_parse(args):
    tok = load_tokenizer()
    for name, fn in (("qaconv", lambda: parse_qaconv(tok, args.budget)),
                     ("friends", lambda: parse_friends(tok, args.budget)),
                     ("squad", parse_squad)):
        if args.source and name != args.source:
            continue
        corpus, qs = fn()
        save_json(corpus, src_path(name, "corpus.json"))
        save_json(qs, src_path(name, "questions.json"))


# ---------------------------------------------------------------- embed (GPU)
def make_item(src, q, corpus, top):
    """The candidate item, or None when the top-50 holds no gold or only gold."""
    gset = set(q["gold"])
    hits = sum(int(ci) in gset for ci in top)
    if hits == 0 or hits == len(top):
        return None
    return {
        "id": f"{src}-{q['qid']}",
        "question": q["question"],
        "n_gold": len(gset),
        "gold_titles": [corpus[g]["cid"] for g in sorted(gset)],
        "paragraphs": [{
            "idx": int(ci),
            "title": corpus[int(ci)].get("title", ""),
            "paragraph_text": corpus[int(ci)]["text"],
            "is_supporting": int(ci) in gset,
        } for ci in top],
    }


def stage_embed(args):
    import numpy as np
    import torch

    src = args.source
    corpus = load_json(src_path(src, "corpus.json"))
    qs = load_json(src_path(src, "questions.json"))
    model = load_embedder(max_seq_length=1024)  # safety cap; no input reaches it
    cemb = model.encode([c["text"] for c in corpus], batch_size=32,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=True).astype(np.float32)
    qemb = model.encode([q["question"] for q in qs], prompt_name="query",
                        batch_size=64, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True
                        ).astype(np.float32)

    out, n_gold_tot, n_in_tot = [], 0, 0
    sims_all = torch.from_numpy(qemb).cuda() @ torch.from_numpy(cemb).cuda().T
    top_all = torch.argsort(sims_all, dim=1, descending=True)[:, :TOPK].cpu().numpy()
    for qi, q in enumerate(qs):
        top = top_all[qi]
        gset = set(q["gold"])
        n_gold_tot += len(gset)
        n_in_tot += sum(int(ci) in gset for ci in top)
        item = make_item(src, q, corpus, top)
        if item is not None:
            out.append(item)
    save_json(out, src_path(src, "candidates.json"))
    print(f"{src}: {len(out)}/{len(qs)} kept "
          f"({100 * len(out) / len(qs):.1f}% retention); "
          f"gold-in-pool R@50 = {100 * n_in_tot / n_gold_tot:.2f}")


# ------------------------------------------------------------------ mix
def stage_mix(args):
    rng = random.Random(0)
    parts = {}
    for name, path in (("musique", work("musique_train", "candidates.json")),
                       ("nqa", work("narrativeqa", "train_candidates.json")),
                       ("qaconv", src_path("qaconv", "candidates.json")),
                       ("friends", src_path("friends", "candidates.json")),
                       ("squad", src_path("squad", "candidates.json"))):
        data = [x for x in load_json(path)
                if 0 < sum(bool(p["is_supporting"]) for p in x["paragraphs"])
                < len(x["paragraphs"])]
        rng.shuffle(data)
        take = min(TARGETS[name], len(data))
        parts[name] = data[:take]
        print(f"{name}: {take} taken (of {len(data)} valid)")
    mixed = [x for v in parts.values() for x in v]
    rng.shuffle(mixed)
    save_json(mixed, mix_path())
    print(f"total: {len(mixed)} -> {mix_path()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["parse", "embed", "mix"])
    ap.add_argument("--source", choices=SOURCES)
    ap.add_argument("--budget", type=int, default=300, help="dialogue chunk budget (Qwen3 tokens)")
    a = ap.parse_args()
    if a.stage == "embed" and not a.source:
        ap.error("embed needs --source")
    {"parse": stage_parse, "embed": stage_embed, "mix": stage_mix}[a.stage](a)


if __name__ == "__main__":
    main()
