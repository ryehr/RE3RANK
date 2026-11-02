"""LoCoMo candidates (QRRanker Table 2 protocol) from locomo10.json (github.com/snap-research/locomo,
fetched by scripts/data/download_raw.sh into <data>/raw/locomo/).

Each of the 10 conversations is chunked on its own: consecutive turns of one session are packed
into chunks of at most --budget Qwen3 tokens (a turn is never split and a chunk never crosses a
session); every chunk starts with "Session date: <date time>", every turn is "speaker: text", and a
turn that shares an image gets " [shares a photo: <BLIP caption>]". Questions: every category
except 5 (adversarial / unanswerable); a question none of whose evidence turns (dia_id) can be
found is dropped (1,531 remain). Candidates: the cosine top-50 chunks of the question's own
conversation (Qwen3-Embedding-8B). Gold: the chunks holding an evidence turn.

One GPU, a few minutes:
  python scripts/data/build_locomo.py      -> paths.EVAL_SETS["locomo"]
"""
import argparse
import ast

import numpy as np

from re3rank import paths

from _data_common import TOPK, load_embedder, load_json, load_tokenizer, raw, save_json


def turn_text(t):
    s = f"{t['speaker']}: {t['text']}"
    cap = t.get("blip_caption")
    if cap:
        s += f" [shares a photo: {cap}]"
    return s


def build_chunks(conv, tok, budget):
    """Returns (chunk_texts, {dia_id: chunk index})."""
    texts, dia2chunk = [], {}
    si = 1
    while f"session_{si}" in conv:
        date = conv.get(f"session_{si}_date_time", "")
        header = f"Session date: {date}\n"
        cur, cur_tok = [], 0

        def flush():
            nonlocal cur, cur_tok
            if cur:
                texts.append(header + "\n".join(turn_text(t) for t in cur))
                for t in cur:
                    dia2chunk[t["dia_id"]] = len(texts) - 1
                cur, cur_tok = [], 0
        for t in conv[f"session_{si}"]:
            n = len(tok(turn_text(t), add_special_tokens=False)["input_ids"])
            if cur and cur_tok + n > budget:
                flush()
            cur.append(t)
            cur_tok += n
        flush()
        si += 1
    return texts, dia2chunk


def eval_questions(sample):
    """The questions kept for evaluation: all but category 5, in file order."""
    return [q for q in sample["qa"] if int(q["category"]) != 5]


def gold_chunks(q, dia2chunk):
    ev = q.get("evidence", [])
    if isinstance(ev, str):
        ev = ast.literal_eval(ev)
    return sorted({dia2chunk[e] for e in ev if e in dia2chunk})


def make_item(sample, qi, q, chunks, top, gset):
    return {
        "id": f"locomo-{sample['sample_id']}-{qi}",
        "question": q["question"],
        "category": int(q["category"]),
        "n_gold": len(gset),
        "gold_titles": [],
        "paragraphs": [{
            "idx": int(ci),
            "title": "",
            "paragraph_text": chunks[int(ci)],
            "is_supporting": int(ci) in gset,
        } for ci in top],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=300, help="chunk budget in Qwen3 tokens")
    ap.add_argument("--src", default=raw("locomo", "locomo10.json"))
    args = ap.parse_args()

    data = load_json(args.src)
    tok = load_tokenizer()
    emb_model = load_embedder()

    out, all_lens = [], []
    n_gold_tot = n_in_tot = n_drop = 0
    for s in data:
        chunks, dia2chunk = build_chunks(s["conversation"], tok, args.budget)
        all_lens += [len(tok(c, add_special_tokens=False)["input_ids"]) for c in chunks]
        cemb = emb_model.encode(chunks, batch_size=32, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False
                                ).astype(np.float32)
        qa = eval_questions(s)
        qemb = emb_model.encode([q["question"] for q in qa], prompt_name="query",
                                batch_size=64, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False
                                ).astype(np.float32)
        for qi, q in enumerate(qa):
            gold = gold_chunks(q, dia2chunk)
            if not gold:
                n_drop += 1
                continue
            sims = cemb @ qemb[qi]
            k = min(TOPK, len(sims))
            top = np.argsort(-sims)[:k]
            gset = set(gold)
            n_gold_tot += len(gold)
            n_in_tot += sum(int(ci) in gset for ci in top)
            out.append(make_item(s, qi, q, chunks, top, gset))
        print(f"{s['sample_id']}: {len(chunks)} chunks, {len(qa)} questions", flush=True)

    path = paths.eval_file("locomo")
    save_json(out, path)
    print(f"\n{len(out)} instances ({n_drop} dropped: no resolvable evidence)")
    print(f"chunk tokens: mean {sum(all_lens)/len(all_lens):.0f}; "
          f"gold chunks/question: {n_gold_tot/len(out):.2f}; "
          f"R@50 ceiling: {100*n_in_tot/n_gold_tot:.2f}")
    print("saved", path)


if __name__ == "__main__":
    main()
