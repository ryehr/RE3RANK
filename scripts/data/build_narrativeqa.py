"""NarrativeQA candidates: the HELMET test set (evaluation) and the silver-labelled training set.

Books and questions come from the HF dataset deepmind/narrativeqa. A book's text is cropped to the
story body with the dataset's start/end markers and cut into non-overlapping 200-token chunks
(Qwen3 tokenizer, sentence boundaries ignored). The candidates of a question are the cosine top-50
chunks of its own book (Qwen3-Embedding-8B, plain question as the query; a book with fewer than 50
chunks gives all of them). Gold chunks are silver labels from the MiA-RAG procedure (Alg. 1):
  1. three queries -- question Q, Q + " " + answer, answer A -- each retrieve their top-10 chunks;
  2. the pooled chunks are ranked by votes (ties: best rank) and the first 10 kept;
  3. an OpenAI model (default gpt-5.4) reads the question, the first reference answer and these 10
     chunks and returns the relevant ones with the MiA-RAG Figure 12 prompt; those are the gold.
Questions without any silver chunk are dropped.

  test   HELMET selection: test-split questions whose raw book is longer than 131,072 Llama-2
         tokens (1,981 questions over 66 books); 79 get no silver chunk, leaving 1,902. Questions
         whose silver chunks all miss the top-50 are kept (they carry n_gold for absolute recall).
         -> paths.EVAL_SETS["narrativeqa"]
  train  20,000 train-split questions drawn with random.Random(42); questions whose top-50 holds
         no silver chunk are dropped. -> work/narrativeqa/train_candidates.json (input of the
         training mixture)

Stages, all under <data>/work/narrativeqa/ (books of both splits share docs.json / chunks.json):
  python scripts/data/build_narrativeqa.py prepare --split test       # CPU (Llama-2 tokenizer)
  python scripts/data/build_narrativeqa.py prepare --split train      # CPU
  python scripts/data/build_narrativeqa.py chunk                      # CPU, chunks every prepared book
  python scripts/data/build_narrativeqa.py embed --split S [--shard i --nshards n]   # GPU
  python scripts/data/build_narrativeqa.py silver --split S           # OpenAI API, key from OPENAI_API_KEY
  python scripts/data/build_narrativeqa.py assemble --split S         # CPU
embed shards split the books (shard 0 also embeds the split's query variants); the silver stage
saves its progress every 500 questions and resumes from it.
"""
import argparse
import os
import random

import numpy as np

from re3rank import paths

from _data_common import TOPK, load_embedder, load_json, load_tokenizer, save_json, work

HF_REPO = "deepmind/narrativeqa"
HF_REVISION = "2e643e7363944af1c33a652d1c87320d0871c4e4"
SPLITS = ("train", "test")
CHUNK_TOKENS = 200
# HELMET (load_narrativeqa): keep test questions whose raw document exceeds this many Llama-2 tokens.
# NousResearch/Llama-2-7b-hf ships the same tokenizer files as the gated meta-llama/Llama-2-7b-hf.
LLAMA2_ID = "NousResearch/Llama-2-7b-hf"
MIN_TOKENS = 131072
TRAIN_N, TRAIN_SEED = 20000, 42
VOTE_K = 10                  # MiA-RAG Appx C.3: top-10 per query variant, 10 voted chunks
DEFAULT_MODEL = "gpt-5.4"


def wpath(name):
    return work("narrativeqa", name)


def out_path(split):
    return paths.eval_file("narrativeqa") if split == "test" else wpath("train_candidates.json")


def crop_story(doc):
    """Crop raw text to the story body using the dataset's start/end markers."""
    text, start, end = doc["text"], doc["start"], doc["end"]
    s = text.find(start) if start else -1
    e = text.rfind(end) if end else -1
    lo = s if s >= 0 else 0
    hi = (e + len(end)) if e >= 0 else len(text)
    if hi <= lo:
        lo, hi = 0, len(text)
    return text[lo:hi]


# ---------------------------------------------------------------- prepare
def load_split(split):
    from datasets import load_dataset
    return load_dataset(HF_REPO, split=split, revision=HF_REVISION)


def llama2_doc_lengths(ds):
    """{doc_id: raw document length in Llama-2 tokens (with BOS)}"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLAMA2_ID)
    doc_len = {}
    for row in ds:
        did = row["document"]["id"]
        if did not in doc_len:
            doc_len[did] = len(tok(row["document"]["text"])["input_ids"])
            if len(doc_len) % 50 == 0:
                print(f"tokenized {len(doc_len)} docs", flush=True)
    return doc_len


def load_docs():
    return load_json(wpath("docs.json")) if os.path.exists(wpath("docs.json")) else {}


def prepare_test():
    ds = load_split("test")
    print("test rows:", len(ds))
    doc_len = llama2_doc_lengths(ds)
    long_docs = {d for d, n in doc_len.items() if n > MIN_TOKENS}
    print(f"docs: {len(doc_len)} total, {len(long_docs)} over {MIN_TOKENS} tokens")

    docs = load_docs()
    questions = []
    for row in ds:
        doc = row["document"]
        if doc["id"] not in long_docs:
            continue
        if doc["id"] not in docs:
            docs[doc["id"]] = {"kind": doc["kind"], "text": crop_story(doc)}
        questions.append({
            "qi": len(questions),
            "doc_id": doc["id"],
            "question": row["question"]["text"],
            "answers": [a["text"] for a in row["answers"]],
        })
    save_json(questions, wpath("questions_test.json"))
    save_json(docs, wpath("docs.json"))
    print(f"{len(questions)} questions over {len(long_docs)} docs")


def prepare_train(n):
    ds = load_split("train")
    print("train rows:", len(ds))
    rng = random.Random(TRAIN_SEED)
    idxs = rng.sample(range(len(ds)), n)

    docs = load_docs()
    questions = []
    for qi, i in enumerate(idxs):
        row = ds[i]
        doc = row["document"]
        did = doc["id"]
        if did not in docs:
            docs[did] = {"kind": doc["kind"], "text": crop_story(doc)}
        questions.append({
            "qi": qi,
            "doc_id": did,
            "question": row["question"]["text"],
            "answers": [a["text"] for a in row["answers"]],
        })
    save_json(questions, wpath("questions_train.json"))
    save_json(docs, wpath("docs.json"))
    print(f"{len(questions)} questions over {len({q['doc_id'] for q in questions})} unique docs")


# ---------------------------------------------------------------- chunk
def chunk_text(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [tok.decode(ids[i:i + CHUNK_TOKENS]).strip()
            for i in range(0, len(ids), CHUNK_TOKENS)]


def stage_chunk():
    tok = load_tokenizer()
    docs = load_json(wpath("docs.json"))
    try:
        chunks = load_json(wpath("chunks.json"))
        print(f"incremental: {len(chunks)} docs already chunked")
    except FileNotFoundError:
        chunks = {}
    todo = [d for d in docs if d not in chunks]
    for n, did in enumerate(todo):
        chunks[did] = chunk_text(tok, docs[did]["text"])
        if (n + 1) % 100 == 0:
            print(f"chunked {n+1}/{len(todo)} docs", flush=True)
    save_json(chunks, wpath("chunks.json"))
    lens = [len(c) for c in chunks.values()]
    print(f"docs: {len(chunks)}, chunks/doc: mean {sum(lens)/len(lens):.0f}, "
          f"min {min(lens)}, max {max(lens)}")


# ---------------------------------------------------------------- embed (GPU)
def stage_embed(split, shard, nshards):
    model = load_embedder()
    os.makedirs(wpath("emb"), exist_ok=True)
    chunks = load_json(wpath("chunks.json"))
    doc_ids = sorted(chunks.keys())[shard::nshards]
    for n, did in enumerate(doc_ids):
        path = wpath(f"emb/{did}.npy")
        if os.path.exists(path):
            continue
        emb = model.encode(chunks[did], batch_size=64, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        np.save(path, emb.astype(np.float16))
        if (n + 1) % 20 == 0:
            print(f"[shard {shard}] embedded {n+1}/{len(doc_ids)} docs", flush=True)

    if shard == 0:
        questions = load_json(wpath(f"questions_{split}.json"))
        variants = []
        for q in questions:
            a = q["answers"][0]
            variants += [q["question"], q["question"] + " " + a, a]
        qemb = model.encode(variants, prompt_name="query", batch_size=64,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)
        np.save(wpath(f"qemb_{split}.npy"), qemb.astype(np.float16))
        print("query variant embeddings saved", flush=True)
    print(f"[shard {shard}] done")


# ---------------------------------------------------------------- silver (OpenAI API)
def vote_topk_candidates(split, k=VOTE_K):
    """MiA-RAG Alg. 1 lines 8-13: pool RetrieveTopK(k) over {Q, Q+A, A}, then VoteAndSelectTopK --
    rank pooled chunks by vote count (best retrieval rank as tiebreak) and keep the top k."""
    questions = load_json(wpath(f"questions_{split}.json"))
    qemb = np.load(wpath(f"qemb_{split}.npy")).astype(np.float32)
    cand = {}
    for q in questions:
        emb = np.load(wpath(f"emb/{q['doc_id']}.npy")).astype(np.float32)
        votes, best_rank = {}, {}
        for v in range(3):
            sims = emb @ qemb[3 * q["qi"] + v]
            kk = min(k, len(sims))
            top = np.argsort(-sims)[:kk]
            for rank, ci in enumerate(top):
                ci = int(ci)
                votes[ci] = votes.get(ci, 0) + 1
                best_rank[ci] = min(best_rank.get(ci, 99), rank)
        ranked = sorted(votes, key=lambda ci: (-votes[ci], best_rank[ci]))
        cand[q["qi"]] = ranked[: min(k, len(ranked))]
    return cand


# Figure 12 of MiA-RAG (arXiv:2512.17220), verbatim.
FILTER_PROMPT_MIARAG = (
    "You are an expert at analyzing narrative texts and selecting relevant passages "
    "to answer questions about stories, novels, and literary works. Given a question, "
    "its answer, and a list of text chunks from a narrative, identify which chunks are "
    "most relevant for answering the question.\n"
    "Input\n"
    "Question: {q}\n"
    "Answer: {a}\n"
    "Text Chunks (indexed from 0): {chunks}\n"
    "Instructions\n"
    "1. Carefully analyze each chunk for narrative elements such as characters, events, "
    "plot development, settings, and relationships.\n"
    "2. Select chunks that:\n"
    "- directly contain information needed to answer the question,\n"
    "- provide essential background context or character development,\n"
    "- describe events or situations relevant to the answer,\n"
    "- include dialogue, actions, or descriptions that inform the question.\n"
    "3. Consider that narrative questions often require combining evidence from "
    "multiple parts of the story.\n"
    "4. Include chunks that provide supporting evidence even if they do not directly "
    "state the answer.\n"
    "5. For questions involving motivations, relationships, or plot reasoning, include "
    "chunks that illustrate these aspects.\n"
    "Output Requirement\n"
    "Return only a JSON array of relevant chunk indices (e.g., [0,2,5]).\n"
    "If none are relevant, return [-1].\n"
    "No explanations or additional text."
)


def stage_silver(split, model_name, concurrency):
    """MiA-RAG Alg. 1 line 15: one list-based filtering call per question (Figure 12 prompt)
    selecting the silver subset of the voted candidates. Progress is saved every 500 questions
    for resume; a question whose call still fails after the retries gets an empty silver set
    (reported at the end)."""
    import asyncio
    import re
    import time
    from openai import AsyncOpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENAI_API_KEY to run the silver stage")
    client = AsyncOpenAI(api_key=key)

    cand = vote_topk_candidates(split)
    questions = load_json(wpath(f"questions_{split}.json"))
    chunks = load_json(wpath("chunks.json"))

    progress = wpath(f"silver_progress_{split}.json")
    try:
        done = {int(k_): v for k_, v in load_json(progress).items()}
        print(f"resuming: {len(done)} questions already done")
    except FileNotFoundError:
        done = {}

    jobs = [q for q in questions if q["qi"] not in done]
    print(f"{len(questions)} questions total, {len(jobs)} to run "
          f"({sum(len(cand[q['qi']]) for q in questions)} voted candidates), model {model_name}")

    usage = {"in": 0, "out": 0, "fail": 0}
    # a parameter the model rejects is dropped and the call retried
    api_kwargs = {"max_completion_tokens": 2048, "reasoning_effort": "none"}

    async def judge(q, sem):
        cis = cand[q["qi"]]
        listing = "\n".join(f"[{i}] {chunks[q['doc_id']][ci]}" for i, ci in enumerate(cis))
        msg = FILTER_PROMPT_MIARAG.format(q=q["question"], a=q["answers"][0],
                                          chunks="\n" + listing)
        async with sem:
            for attempt in range(6):
                try:
                    r = await client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": msg}],
                        **api_kwargs)
                    usage["in"] += r.usage.prompt_tokens
                    usage["out"] += r.usage.completion_tokens
                    txt = (r.choices[0].message.content or "")
                    m = re.search(r"\[[-0-9,\s]*\]", txt)
                    if m is None:
                        raise ValueError(f"unparsable: {txt[:80]}")
                    idxs = [int(x) for x in re.findall(r"-?\d+", m.group(0))]
                    keep = [cis[i] for i in idxs if 0 <= i < len(cis)]
                    return q["qi"], keep
                except Exception as e:
                    err = str(e)
                    changed = False
                    for p in list(api_kwargs):
                        if f"'{p}'" in err:
                            api_kwargs.pop(p)
                            changed = True
                    if not changed:
                        await asyncio.sleep(min(60, 2 ** attempt))
            usage["fail"] += 1
            return q["qi"], []

    def save_progress():
        save_json({str(k_): v for k_, v in done.items()}, progress)

    async def run():
        sem = asyncio.Semaphore(concurrency)
        tasks = [asyncio.create_task(judge(q, sem)) for q in jobs]
        t0, n = time.time(), 0
        for fut in asyncio.as_completed(tasks):
            qi, keep = await fut
            done[qi] = keep
            n += 1
            if n % 500 == 0:
                save_progress()
                el = time.time() - t0
                print(f"[{n}/{len(jobs)}] {n/el:.1f} req/s  tokens in/out: "
                      f"{usage['in']}/{usage['out']}  fails: {usage['fail']}", flush=True)
        save_progress()

    asyncio.run(run())

    silver = {str(q["qi"]): done.get(q["qi"], []) for q in questions}
    kept = sum(len(v) for v in silver.values())
    save_json(silver, wpath(f"silver_{split}.json"))
    print(f"avg silver chunks/question: {kept/len(questions):.2f} (MiA-RAG ~2.3); "
          f"api failures: {usage['fail']}; tokens in/out: {usage['in']}/{usage['out']}")


# ---------------------------------------------------------------- assemble
def make_item(split, q, top, sil, book_chunks):
    item = {"id": f"nqa-{q['qi']}", "question": q["question"]}
    if split == "test":
        item["n_gold"] = len(sil)
    item["gold_titles"] = []
    item["paragraphs"] = [{
        "idx": int(ci),
        "title": "",
        "paragraph_text": book_chunks[int(ci)],
        "is_supporting": int(ci) in sil,
    } for ci in top]
    return item


def stage_assemble(split):
    questions = load_json(wpath(f"questions_{split}.json"))
    chunks = load_json(wpath("chunks.json"))
    silver = load_json(wpath(f"silver_{split}.json"))
    qemb = np.load(wpath(f"qemb_{split}.npy")).astype(np.float32)

    out, skipped = [], 0
    for q in questions:
        sil = set(silver[str(q["qi"])])
        if not sil:
            skipped += 1
            continue
        emb = np.load(wpath(f"emb/{q['doc_id']}.npy")).astype(np.float32)
        sims = emb @ qemb[3 * q["qi"]]  # plain-question variant
        k = min(TOPK, len(sims))
        top = np.argsort(-sims)[:k]
        if split == "train" and not any(int(ci) in sil for ci in top):
            skipped += 1
            continue
        out.append(make_item(split, q, top, sil, chunks[q["doc_id"]]))
    path = out_path(split)
    save_json(out, path)
    golds = [sum(p["is_supporting"] for p in s["paragraphs"]) for s in out]
    msg = (f"{len(out)} instances ({skipped} skipped); "
           f"positives-in-candidates/instance: {sum(golds)/len(golds):.2f}")
    if split == "test":
        msg += f"; R@50 ceiling: {100*sum(golds)/sum(s['n_gold'] for s in out):.2f}"
    print(msg)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "chunk", "embed", "silver", "assemble"])
    ap.add_argument("--split", choices=SPLITS, help="required by every stage but chunk")
    ap.add_argument("--n", type=int, default=TRAIN_N, help="prepare --split train: questions drawn")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="silver: OpenAI model")
    ap.add_argument("--concurrency", type=int, default=24, help="silver: parallel API calls")
    args = ap.parse_args()
    if args.stage != "chunk" and args.split is None:
        ap.error(f"{args.stage} needs --split")
    if args.stage == "prepare":
        prepare_test() if args.split == "test" else prepare_train(args.n)
    elif args.stage == "chunk":
        stage_chunk()
    elif args.stage == "embed":
        stage_embed(args.split, args.shard, args.nshards)
    elif args.stage == "silver":
        stage_silver(args.split, args.model, args.concurrency)
    else:
        stage_assemble(args.split)


if __name__ == "__main__":
    main()
