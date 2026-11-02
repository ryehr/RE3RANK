"""Candidates for the five Wikipedia QA sets of the HippoRAG 2 release (HF dataset osunlp/HippoRAG_2).

Every set has 1,000 questions and one passage corpus shared by all of them; a passage is one
candidate (no further chunking). Qwen3-Embedding-8B encodes each passage as "title\\ntext" and each
question with the model's "query" prompt; the cosine top-50 over the whole corpus is kept in
retrieval order.

  set       HippoRAG 2 files                    gold passages
  hotpotqa  hotpotqa(.json, _corpus.json)        passages whose title is a supporting_facts title
  2wiki     2wikimultihopqa(...)                 the context paragraphs of the supporting_facts titles
  musique   musique(...)                         the paragraphs marked is_supporting
  nq        nq_rear(...)                         the contexts marked is_supporting
  popqa     popqa(...)                           the paragraphs marked is_supporting

HotpotQA uses the corpus as released and labels gold by title (each candidate also keeps its
retriever_score). The other four deduplicate the corpus by (title, text) first and label gold by
exact (title, text) match (MuSiQue and PopQA repeat titles, so a title match would mislabel); each
item records n_gold, the number of gold passages of the question.

GPU (embedding), a few minutes per set:
  python scripts/data/build_hipporag.py --dataset all
Writes paths.EVAL_SETS[set] for each set.
"""
import argparse

from re3rank import paths

from _data_common import EMB_ID, TOPK, hf_file, load_embedder, load_json, save_json

HF_REPO = "osunlp/HippoRAG_2"
HF_REVISION = "5ec05b38deecc3318bb432c69865959c56058990"
# evaluation set -> file prefix in the HippoRAG 2 release
FILES = {"hotpotqa": "hotpotqa", "2wiki": "2wikimultihopqa", "musique": "musique",
         "nq": "nq_rear", "popqa": "popqa"}


def fetch(name):
    """(questions, corpus) as released."""
    prefix = FILES[name]
    data = load_json(hf_file(HF_REPO, f"{prefix}.json", HF_REVISION))
    corpus = load_json(hf_file(HF_REPO, f"{prefix}_corpus.json", HF_REVISION))
    return data, corpus


def prepare_matched(name, data, corpus):
    """Questions [{id, question, golds: {(title, text), ...}}] and the (title, text)-deduplicated
    corpus, for the four sets labelled by exact passage match."""
    questions = []
    for i, q in enumerate(data):
        if name == "nq":
            golds = {(c["title"], c["text"]) for c in q["contexts"] if c["is_supporting"]}
            qid = f"nq-{i}"
        elif name == "popqa":
            golds = {(p["title"], p["text"]) for p in q["paragraphs"] if p["is_supporting"]}
            qid = str(q.get("id", f"popqa-{i}"))
        elif name == "musique":
            golds = {(p["title"], p["paragraph_text"]) for p in q["paragraphs"] if p["is_supporting"]}
            qid = q["id"]
        elif name == "2wiki":
            sup_titles = {t for t, _ in q["supporting_facts"]}
            golds = {(title, " ".join(sents)) for title, sents in q["context"]
                     if title in sup_titles}
            qid = q["_id"]
        else:
            raise ValueError(name)
        questions.append({"id": qid, "question": q["question"], "golds": golds})

    seen, dedup = set(), []
    for c in corpus:
        key = (c["title"], c["text"])
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    return questions, dedup


def retrieve(corpus, question_texts, k, batch):
    """torch.topk result (indices, values) of question-passage cosine similarity."""
    import torch
    model = load_embedder()
    doc_texts = [f"{c['title']}\n{c['text']}" for c in corpus]
    doc_emb = model.encode(doc_texts, batch_size=batch, normalize_embeddings=True,
                           convert_to_tensor=True, show_progress_bar=True)
    q_emb = model.encode(question_texts, prompt_name="query", batch_size=batch,
                         normalize_embeddings=True, convert_to_tensor=True,
                         show_progress_bar=False)
    sims = q_emb @ doc_emb.T
    return torch.topk(sims, k=k, dim=1)


def assemble_hotpotqa(questions, corpus, topk):
    out, recalls = [], {k: [] for k in (3, 5, 10, 20, 50)}
    for qi, q in enumerate(questions):
        gold_titles = {sf[0] for sf in q["supporting_facts"]}
        paragraphs = []
        for rank, di in enumerate(topk.indices[qi].tolist()):
            d = corpus[di]
            paragraphs.append({
                "idx": d["idx"],
                "title": d["title"],
                "paragraph_text": d["text"],
                "is_supporting": d["title"] in gold_titles,
                "retriever_score": float(topk.values[qi][rank]),
            })
        for k in recalls:
            found = {p["title"] for p in paragraphs[:k] if p["is_supporting"]}
            recalls[k].append(len(found) / len(gold_titles))
        out.append({
            "id": q["_id"],
            "question": q["question"],
            "answer": q.get("answer"),
            "gold_titles": sorted(gold_titles),
            "paragraphs": paragraphs,
        })
    for k, v in recalls.items():
        print(f"retriever R@{k}: {100 * sum(v) / len(v):.2f}")
    return out


def assemble_matched(questions, corpus, top_idx):
    out = []
    for q, idxs in zip(questions, top_idx):
        paragraphs = []
        for ci in idxs:
            c = corpus[ci]
            paragraphs.append({
                "idx": ci,
                "title": c["title"],
                "paragraph_text": c["text"],
                "is_supporting": (c["title"], c["text"]) in q["golds"],
            })
        out.append({
            "id": q["id"],
            "question": q["question"],
            "gold_titles": sorted({t for t, _ in q["golds"]}),
            "n_gold": len(q["golds"]),
            "paragraphs": paragraphs,
        })
    hit = [sum(p["is_supporting"] for p in o["paragraphs"]) / o["n_gold"] for o in out]
    print(f"R@50 (gold coverage in candidates): {100 * sum(hit) / len(hit):.2f}")
    return out


def build(name, args):
    data, corpus = fetch(name)
    if name == "hotpotqa":
        print(f"{name}: {len(data)} questions, {len(corpus)} passages")
        topk = retrieve(corpus, [q["question"] for q in data], args.topk, args.batch)
        out = assemble_hotpotqa(data, corpus, topk)
    else:
        questions, corpus = prepare_matched(name, data, corpus)
        print(f"{name}: {len(questions)} questions, corpus {len(corpus)} (deduplicated)")
        gold_keys_all = set().union(*(q["golds"] for q in questions))
        corpus_keys = {(c["title"], c["text"]) for c in corpus}
        print(f"gold passages: {len(gold_keys_all)}, missing from corpus: "
              f"{len(gold_keys_all - corpus_keys)}")
        topk = retrieve(corpus, [q["question"] for q in questions],
                        min(args.topk, len(corpus)), args.batch)
        out = assemble_matched(questions, corpus, topk.indices.cpu().tolist())
    path = paths.eval_file(name)
    save_json(out, path)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(FILES) + ["all"])
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--batch", type=int, default=32, help="embedding batch size")
    args = ap.parse_args()
    print(f"retriever: {EMB_ID}")
    for name in (list(FILES) if args.dataset == "all" else [args.dataset]):
        build(name, args)


if __name__ == "__main__":
    main()
