"""MuSiQue-train candidates (training source) from the HF dataset dgslibisey/MuSiQue (answerable
train split, 19,938 questions).

Pool: the paragraphs of all train questions, deduplicated by (title, text) into one corpus shared
by every question (84,559 paragraphs); a paragraph is one candidate. Qwen3-Embedding-8B encodes
"title\\ntext" and the question with the "query" prompt; the cosine top-50 is kept. Gold: the
candidates equal (title and text) to one of the question's official supporting paragraphs.
Questions whose top-50 holds no gold are dropped.

Stages, under <data>/work/musique_train/:
  python scripts/data/build_musique_train.py prep                          # CPU
  python scripts/data/build_musique_train.py embed --shard i --nshards 3   # GPU, one per shard
  python scripts/data/build_musique_train.py assemble --nshards 3          # GPU (top-k)
-> work/musique_train/candidates.json (input of the training mixture). embed shards split the
corpus round-robin (shard 0 also embeds the questions); the released data used --nshards 3.
"""
import argparse
import os

import numpy as np

from _data_common import TOPK, load_embedder, load_json, save_json, work

HF_REPO = "dgslibisey/MuSiQue"
HF_REVISION = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"


def wpath(name):
    return work("musique_train", name)


def out_path():
    return wpath("candidates.json")


def stage_prep():
    from datasets import load_dataset
    os.makedirs(wpath(""), exist_ok=True)
    ds = load_dataset(HF_REPO, split="train", revision=HF_REVISION)
    print("rows:", len(ds), "fields:", ds.column_names)

    questions, corpus, seen = [], [], {}
    for row in ds:
        golds = []
        for p in row["paragraphs"]:
            key = (p["title"], p["paragraph_text"])
            if key not in seen:
                seen[key] = len(corpus)
                corpus.append({"title": p["title"], "text": p["paragraph_text"]})
            if p["is_supporting"]:
                golds.append(list(key))
        questions.append({"id": row["id"], "question": row["question"], "golds": golds})

    save_json(questions, wpath("questions.json"))
    save_json(corpus, wpath("corpus.json"))
    ng = [len(q["golds"]) for q in questions]
    print(f"{len(questions)} questions, corpus {len(corpus)} unique paragraphs, "
          f"golds/question mean {sum(ng)/len(ng):.2f}")


def stage_embed(shard, nshards):
    model = load_embedder()
    corpus = load_json(wpath("corpus.json"))
    idxs = list(range(shard, len(corpus), nshards))
    texts = [f"{corpus[i]['title']}\n{corpus[i]['text']}" for i in idxs]
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    np.save(wpath(f"emb_corpus.{shard}.npy"), emb.astype(np.float16))
    print(f"shard {shard}: {len(texts)} paragraphs embedded")

    if shard == 0:
        questions = load_json(wpath("questions.json"))
        qemb = model.encode([q["question"] for q in questions], prompt_name="query",
                            batch_size=64, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=True)
        np.save(wpath("emb_query.npy"), qemb.astype(np.float16))
        print("queries embedded")


def make_item(q, corpus, idxs):
    """(item or None if no gold among the candidates, gold coverage of the question)"""
    gold_keys = {tuple(g) for g in q["golds"]}
    paragraphs = [{
        "idx": ci,
        "title": corpus[ci]["title"],
        "paragraph_text": corpus[ci]["text"],
        "is_supporting": (corpus[ci]["title"], corpus[ci]["text"]) in gold_keys,
    } for ci in idxs]
    npos = sum(p["is_supporting"] for p in paragraphs)
    cover = npos / len(gold_keys)
    if npos == 0:
        return None, cover
    return {"id": q["id"], "question": q["question"],
            "gold_titles": sorted({t for t, _ in gold_keys}),
            "n_gold": len(gold_keys), "paragraphs": paragraphs}, cover


def stage_assemble(nshards):
    import torch
    corpus = load_json(wpath("corpus.json"))
    questions = load_json(wpath("questions.json"))

    dim = np.load(wpath("emb_corpus.0.npy")).shape[1]
    doc_emb = np.zeros((len(corpus), dim), dtype=np.float16)
    for s in range(nshards):
        idxs = list(range(s, len(corpus), nshards))
        doc_emb[idxs] = np.load(wpath(f"emb_corpus.{s}.npy"))
    q_emb = np.load(wpath("emb_query.npy"))

    device = "cuda"
    D = torch.from_numpy(doc_emb).to(device, torch.float32)
    out, dropped, cover = [], 0, []
    B = 512
    for lo in range(0, len(questions), B):
        Q = torch.from_numpy(q_emb[lo:lo + B]).to(device, torch.float32)
        top = torch.topk(Q @ D.T, k=TOPK, dim=1).indices.cpu().tolist()
        for qi, idxs in enumerate(top):
            item, c = make_item(questions[lo + qi], corpus, idxs)
            cover.append(c)
            if item is None:
                dropped += 1
                continue
            out.append(item)
    save_json(out, out_path())
    print(f"R@{TOPK} coverage: {100*sum(cover)/len(cover):.2f}")
    print(f"{len(out)} instances kept, {dropped} dropped (no gold in top-{TOPK}, "
          f"{100*dropped/len(questions):.1f}%)")
    print("saved", out_path())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "embed", "assemble"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=3)
    args = ap.parse_args()
    if args.stage == "prep":
        stage_prep()
    elif args.stage == "embed":
        stage_embed(args.shard, args.nshards)
    else:
        stage_assemble(args.nshards)


if __name__ == "__main__":
    main()
