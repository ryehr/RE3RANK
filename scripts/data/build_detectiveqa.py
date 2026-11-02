"""DetectiveQA (English) candidates from the HF dataset Phospheneser/DetectiveQA.

Questions: all 600 English questions (154 from human_anno, 446 from AIsup_anno) that have at
least one valid clue paragraph (clue_position); the question stem only, without the options.
Each novel (62) is numbered by paragraph; consecutive whole paragraphs are packed into chunks of at
most 200 Qwen3 tokens (a paragraph is never split, an overlong one is a chunk of its own), so every
clue paragraph lies in exactly one chunk. Candidates: the cosine top-50 chunks of the question's
own novel (Qwen3-Embedding-8B). Gold: the chunks holding a clue paragraph.

Stages, under <data>/work/detectiveqa/:
  python scripts/data/build_detectiveqa.py prep        # CPU
  python scripts/data/build_detectiveqa.py chunk       # CPU
  python scripts/data/build_detectiveqa.py embed [--shard i --nshards n]   # GPU
  python scripts/data/build_detectiveqa.py assemble    # CPU -> paths.EVAL_SETS["detectiveqa"]
"""
import argparse
import glob
import os
import re

import numpy as np

from re3rank import paths

from _data_common import (TOPK, hf_snapshot, load_embedder, load_json, load_tokenizer,
                          save_json, work)

HF_REPO = "Phospheneser/DetectiveQA"
HF_REVISION = "001e5c590ddeb6c83083c55fa10b0b8b3dcba2ed"
LANG = "en"
CHUNK_TOKENS = 200


def wpath(name):
    return work("detectiveqa", name)


def stage_prep():
    snap = hf_snapshot(HF_REPO, HF_REVISION,
                       allow_patterns=[f"novel_data_{LANG}/*", f"anno_data_{LANG}/*"])
    paras = {}
    for f in glob.glob(f"{snap}/novel_data_{LANG}/*.txt"):
        nid = os.path.basename(f).split("-")[0]
        plist = {}
        for line in open(f).read().splitlines():
            m = re.match(r"\[(\d+)\]\s?(.*)", line)
            if m:
                plist[int(m.group(1))] = m.group(2).strip()
        paras[nid] = plist

    questions = []
    for src in ["human_anno", "AIsup_anno"]:
        for f in sorted(glob.glob(f"{snap}/anno_data_{LANG}/{src}/*.json")):
            nid = os.path.basename(f)[:-5]
            data = load_json(f)
            for item in (data if isinstance(data, list) else [data]):
                for q in item["questions"]:
                    clues = sorted({c for c in q.get("clue_position", [])
                                    if isinstance(c, int) and c >= 0})
                    clues = [c for c in clues if c in paras[nid]]
                    if not clues:
                        continue
                    questions.append({
                        "qi": len(questions),
                        "novel_id": nid,
                        "src": src,
                        "question": q["question"].strip(),
                        "clue_paras": clues,
                    })
    save_json({k: {str(i): t for i, t in v.items()} for k, v in paras.items()},
              wpath(f"paras_{LANG}.json"))
    save_json(questions, wpath(f"questions_{LANG}.json"))
    novels = {q["novel_id"] for q in questions}
    print(f"{len(questions)} questions over {len(novels)} novels; clue paras/question: "
          f"{sum(len(q['clue_paras']) for q in questions)/len(questions):.2f}")


def pack_paragraphs(tok, plist):
    """Whole paragraphs, in paragraph-number order, packed into chunks of <= CHUNK_TOKENS tokens.
    Returns (chunk_texts, {paragraph number (str): chunk index})."""
    order = sorted(plist, key=int)
    texts, p2c, cur, cur_tok = [], {}, [], 0
    for pk in order:
        t = plist[pk]
        n = len(tok(t, add_special_tokens=False)["input_ids"])
        if cur and cur_tok + n > CHUNK_TOKENS:
            texts.append("\n".join(x[1] for x in cur))
            for xk, _ in cur:
                p2c[xk] = len(texts) - 1
            cur, cur_tok = [], 0
        cur.append((pk, t))
        cur_tok += n
    if cur:
        texts.append("\n".join(x[1] for x in cur))
        for xk, _ in cur:
            p2c[xk] = len(texts) - 1
    return texts, p2c


def stage_chunk():
    tok = load_tokenizer()
    paras = load_json(wpath(f"paras_{LANG}.json"))
    out = {}
    for nid, plist in paras.items():
        texts, p2c = pack_paragraphs(tok, plist)
        out[nid] = {"chunks": texts, "p2c": p2c}
    save_json(out, wpath(f"chunks_{LANG}.json"))
    lens = [len(v["chunks"]) for v in out.values()]
    print(f"{len(out)} novels, chunks/novel: mean {sum(lens)/len(lens):.0f}, "
          f"min {min(lens)}, max {max(lens)}")


def stage_embed(shard, nshards):
    model = load_embedder()
    os.makedirs(wpath(f"emb_{LANG}"), exist_ok=True)
    chunks = load_json(wpath(f"chunks_{LANG}.json"))
    for n, nid in enumerate(sorted(chunks.keys())[shard::nshards]):
        path = wpath(f"emb_{LANG}/{nid}.npy")
        if os.path.exists(path):
            continue
        emb = model.encode(chunks[nid]["chunks"], batch_size=32,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)
        np.save(path, emb.astype(np.float16))
        print(f"[shard {shard}] {n+1} novels embedded", flush=True)

    if shard == 0:
        questions = load_json(wpath(f"questions_{LANG}.json"))
        qemb = model.encode([q["question"] for q in questions],
                            prompt_name="query", batch_size=64,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)
        np.save(wpath(f"qemb_{LANG}.npy"), qemb.astype(np.float16))
        print("query embeddings saved", flush=True)


def make_item(q, nv, top, gset):
    return {
        "id": f"detqa-{LANG}-{q['qi']}",
        "question": q["question"],
        "n_gold": len(gset),
        "gold_titles": [],
        "paragraphs": [{
            "idx": int(ci),
            "title": "",
            "paragraph_text": nv["chunks"][int(ci)],
            "is_supporting": int(ci) in gset,
        } for ci in top],
    }


def gold_chunks(q, nv):
    return sorted({nv["p2c"][str(p)] for p in q["clue_paras"]})


def stage_assemble():
    questions = load_json(wpath(f"questions_{LANG}.json"))
    chunks = load_json(wpath(f"chunks_{LANG}.json"))
    qemb = np.load(wpath(f"qemb_{LANG}.npy")).astype(np.float32)

    out = []
    n_gold_tot = n_in_tot = 0
    for q in questions:
        nv = chunks[q["novel_id"]]
        gold = gold_chunks(q, nv)
        emb = np.load(wpath(f"emb_{LANG}/{q['novel_id']}.npy")).astype(np.float32)
        sims = emb @ qemb[q["qi"]]
        k = min(TOPK, len(sims))
        top = np.argsort(-sims)[:k]
        gset = set(gold)
        n_gold_tot += len(gold)
        n_in_tot += sum(int(ci) in gset for ci in top)
        out.append(make_item(q, nv, top, gset))
    path = paths.eval_file("detectiveqa")
    save_json(out, path)
    print(f"{len(out)} instances; gold chunks/question: {n_gold_tot/len(out):.2f}; "
          f"R@50 ceiling: {100*n_in_tot/n_gold_tot:.2f}")
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "chunk", "embed", "assemble"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.stage == "prep":
        stage_prep()
    elif args.stage == "chunk":
        stage_chunk()
    elif args.stage == "embed":
        stage_embed(args.shard, args.nshards)
    else:
        stage_assemble()


if __name__ == "__main__":
    main()
