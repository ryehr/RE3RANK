"""LoCoMo downstream QA under the QRRanker paper's Table 3 protocol: rerank the 50 first-stage chunks, keep only the
top-3 as context, answer with an OpenAI model using the paper's LoCoMo QA prompt (App. J.4, Mem0's answer prompt,
sent as one system message at temperature 0 the way Mem0's evaluation does; reasoning models (the GPT-5 family, gpt-6-luna)
take no temperature and run at their default reasoning effort), and score token F1 per LoCoMo category:
  single-hop = category 4, multi-hop = 1, temporal = 2, open-domain = 3 (5 = adversarial, not in the set);
  Overall = question-weighted mean (reproduces the paper's Overall from its per-category F1 and the 841/282/321/96 counts).
Two F1s are reported:
  locomo  the LoCoMo repo's evaluation: normalize (lower, strip punctuation / commas, drop a|an|the|and), Porter stems,
          token-multiset F1; multi-hop gold and prediction split on commas (mean over gold parts of the best-matching
          prediction part); open-domain gold cut at the first ';'
  simple  Mem0's evaluation: lower, . , ! ? -> space, set-of-tokens F1
Questions: the 1531 of the LoCoMo candidate file (paths.eval_file("locomo"); the 1540 non-adversarial LoCoMo-10
questions minus 9 whose evidence turns do not resolve); gold answers are read back from the raw LoCoMo-10 file (--raw)
by the builder's id (sample id + index in the non-adversarial qa list), with the question text checked.

Ranking sources, --system NAME=SPEC (NAME names the system in the tables):
  first              the first-stage (Qwen3-Embedding-8B) order, i.e. the candidate file's order
  scores:<file>      per-item scores, best = highest, ties keep the first-stage order: either {id: [50 scores]}
                     (scripts/baselines/rerank_score.py) or {id: {"s": [50 scores], ...}} (scripts/eval/score.py
                     --keep-nogold --out, the RE3Rank scorer); a directory means <dir>/locomo.json
  order:<file>       {id: [candidate positions, best first]} (scripts/baselines/qrranker_score.py --order-out)
Every question needs a ranking, including the ones without a gold chunk in the pool (hence --keep-nogold).
Replies go to <qa-dir>/<llm>/<NAME>.jsonl, one line per question, appended as they arrive (resumable, usage recorded);
<qa-dir> defaults to $RE3RANK_OUT/downstream/locomo_qa. The run stage reads the API key from OPENAI_API_KEY.

usage: python scripts/downstream/locomo_qa.py run --llm gpt-4o-mini --system first=first [--limit 20] [--workers 16]
       python scripts/downstream/locomo_qa.py report [--llm ...]
       python scripts/downstream/locomo_qa.py sig [--base qrranker]   (paired per-question test of every system vs
                                                                       --base, per LLM)
"""
import argparse
import collections
import glob
import json
import os
import re
import string
import sys
import threading
import time

from re3rank import paths

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.eval_file("locomo")
RAW = paths.data("raw", "locomo", "locomo10.json")       # the LoCoMo-10 release file (--raw)
QA_DIR = paths.out("downstream", "locomo_qa")             # --qa-dir
TOPK = 3
CATS = [(4, "Single-hop"), (1, "Multi-hop"), (2, "Temporal"), (3, "Open-domain")]

PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers.
2. Pay special attention to the timestamps to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. If the memories contain contradictory information, prioritize the most recent memory.
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

APPROACH (THINK STEP BY STEP):
1. First, examine all memories that contain information related to the question.
2. Examine the timestamps and content of these memories carefully.
3. Look for explicit mentions of dates, times, locations, or events that answer the question.
4. If the answer requires calculation (e.g., converting relative time references), show your work.
5. Formulate a precise, concise answer based solely on the evidence in the memories.
6. Double-check that your answer directly addresses the question asked.
7. Ensure your final answer is specific and avoids vague time references.

RELEVANT MEMORIES:
{context}

QUESTION: {question}
ANSWER:"""


# ------------------------------------------------------------------ data and rankings
def load_items():
    items = json.load(open(DATA))
    raw = {s["sample_id"]: [q for q in s["qa"] if int(q["category"]) != 5] for s in json.load(open(RAW))}
    for it in items:
        sid, qi = it["id"][len("locomo-"):].rsplit("-", 1)
        q = raw[sid][int(qi)]
        assert q["question"] == it["question"], it["id"]
        it["answer"] = str(q["answer"])
    return items


def ranking(spec, items):
    """-> {id: [candidate positions, best first]}"""
    if spec == "first":
        return {it["id"]: list(range(len(it["paragraphs"]))) for it in items}
    kind, arg = spec.split(":", 1)
    if kind == "order":
        return json.load(open(arg))
    if kind != "scores":
        raise ValueError(f"unknown ranking source {spec!r} (first | scores:<file> | order:<file>)")
    S = json.load(open(os.path.join(arg, "locomo.json") if os.path.isdir(arg) else arg))
    out = {}
    for it in items:
        s = S[it["id"]]["s"] if isinstance(S[it["id"]], dict) else S[it["id"]]
        out[it["id"]] = sorted(range(len(s)), key=lambda i: (-s[i], i))
    return out


# ------------------------------------------------------------------ answer extraction and F1
def extract(text):
    """The model is asked to think step by step but to end on a <=6-word answer: keep what follows the last
    'answer:' marker if there is one, else the whole reply (Mem0 scores the whole reply)."""
    t = (text or "").strip()
    m = list(re.finditer(r"(?i)\banswer\s*:", t))
    if m:
        t = t[m[-1].end():].strip()
    return t.strip().strip("*").strip()


_stem = None
def _ps(w):
    global _stem
    if _stem is None:
        from nltk.stem import PorterStemmer
        _stem = PorterStemmer()
    return _stem.stem(w)


def _norm(s):
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def _f1_tokens(pred, gold):
    p = [_ps(w) for w in _norm(pred).split()]
    g = [_ps(w) for w in _norm(gold).split()]
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def f1_locomo(pred, gold, cat):
    if cat == 3:
        gold = gold.split(";")[0].strip()
    if cat == 1:
        preds = [x.strip() for x in pred.split(",")]
        golds = [x.strip() for x in gold.split(",")]
        return sum(max(_f1_tokens(p, g) for p in preds) for g in golds) / len(golds)
    return _f1_tokens(pred, gold)


def f1_simple(pred, gold):
    tok = lambda s: set(str(s).lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split())
    p, g = tok(pred), tok(gold)
    c = p & g
    if not p or not g or not c:
        return 0.0
    pr, rc = len(c) / len(p), len(c) / len(g)
    return 2 * pr * rc / (pr + rc)


# ------------------------------------------------------------------ run
def call(client, llm, prompt, max_out, retries=8):
    kw = dict(model=llm, messages=[{"role": "system", "content": prompt}], max_completion_tokens=max_out)
    if llm.startswith(("gpt-4o", "gpt-4.1", "gpt-3.5")):            # reasoning models only take the default (1)
        kw["temperature"] = 0.0
    for a in range(retries):
        try:
            r = client.chat.completions.create(**kw)
            u = r.usage
            det = getattr(u, "completion_tokens_details", None)
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            return r.choices[0].message.content or "", dict(
                prompt=u.prompt_tokens, completion=u.completion_tokens, cached=cached,
                reasoning=(getattr(det, "reasoning_tokens", 0) or 0) if det else 0,
                finish=r.choices[0].finish_reason)
        except Exception as e:                                   # rate limits, timeouts, 5xx: back off and retry
            if a == retries - 1 or getattr(e, "status_code", None) in (400, 401, 403, 404):
                raise
            time.sleep(min(60, 2 ** a) + 0.1 * a)


def stage_run(args):
    import tiktoken
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        sys.exit("the run stage needs the OpenAI API key in the environment variable OPENAI_API_KEY")
    client = OpenAI(api_key=key, timeout=300)
    enc = tiktoken.get_encoding("o200k_base")
    items = load_items()
    if args.limit:
        items = items[: args.limit]
    for sysarg in args.system:
        name, spec = sysarg.split("=", 1)
        order = ranking(spec, items)
        out = os.path.join(QA_DIR, args.llm, f"{name}.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        done = set()
        if os.path.exists(out):
            done = {json.loads(l)["id"] for l in open(out) if l.strip()}
        todo = [it for it in items if it["id"] not in done]
        print(f"[{args.llm} | {name}] {len(todo)} to do ({len(done)} done), ranking {spec}", flush=True)
        lock, n = threading.Lock(), [0]

        def work(it):
            top = order[it["id"]][:TOPK]
            ctx = "\n\n".join(it["paragraphs"][i]["paragraph_text"] for i in top)
            text, usage = call(client, args.llm, PROMPT.format(context=ctx, question=it["question"]), args.max_out)
            rec = dict(id=it["id"], category=it["category"], question=it["question"], answer=it["answer"],
                       top=[it["paragraphs"][i]["idx"] for i in top],
                       gold_in_top=sum(bool(it["paragraphs"][i]["is_supporting"]) for i in top),
                       ctx_tokens=len(enc.encode(ctx)), raw=text, usage=usage)
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n[0] += 1
                if n[0] % 100 == 0:
                    print(f"  {n[0]}/{len(todo)}", flush=True)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(work, todo))
        print(f"[{args.llm} | {name}] done", flush=True)


# ------------------------------------------------------------------ report
def summarize(path):
    R = [json.loads(l) for l in open(path) if l.strip()]
    by = collections.defaultdict(lambda: [[], []])
    for r in R:
        p = extract(r["raw"])
        by[r["category"]][0].append(f1_locomo(p, r["answer"], r["category"]))
        by[r["category"]][1].append(f1_simple(p, r["answer"]))
    res = {"n": len(R), "tokens": sum(r["ctx_tokens"] for r in R) / len(R),
           "gold_in_top": sum(r["gold_in_top"] > 0 for r in R) / len(R),
           "prompt": sum(r["usage"]["prompt"] for r in R), "completion": sum(r["usage"]["completion"] for r in R),
           "reasoning": sum(r["usage"]["reasoning"] for r in R),
           "truncated": sum(r["usage"]["finish"] == "length" for r in R)}
    for j, key in enumerate(("locomo", "simple")):
        res[key] = {c: 100 * sum(by[c][j]) / len(by[c][j]) for c, _ in CATS if by[c][j]}
        res[key]["overall"] = 100 * sum(sum(by[c][j]) for c in by) / len(R)
    res["n_cat"] = {c: len(by[c][0]) for c, _ in CATS}
    return res


def stage_report(args):
    rows = []
    for d in sorted(glob.glob(os.path.join(QA_DIR, "*"))):
        llm = os.path.basename(d)
        if args.llm and llm not in args.llm:
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            rows.append((llm, os.path.basename(f)[:-6], summarize(f)))
    L = []
    for key, title in (("locomo", "F1, LoCoMo evaluation (Porter stems; multi-hop split on commas)"),
                       ("simple", "F1, Mem0 evaluation (set-of-tokens)")):
        L += [f"### {title}", "",
              "| LLM | Method | n | Tokens | " + " | ".join(c for _, c in CATS) + " | Overall F1 |",
              "|---|---|---|---|" + "---|" * (len(CATS) + 1)]
        for llm, name, r in rows:
            L.append(f"| {llm} | {name} | {r['n']} | {r['tokens']:.0f} | "
                     + " | ".join(f"{r[key].get(c, float('nan')):.2f}" for c, _ in CATS) + f" | {r[key]['overall']:.2f} |")
        L.append("")
    L += ["### usage", "", "| LLM | Method | top-3 holds a gold | prompt tok | completion tok | reasoning tok | cut at max |",
          "|---|---|---|---|---|---|---|"]
    for llm, name, r in rows:
        L.append(f"| {llm} | {name} | {100 * r['gold_in_top']:.1f}% | {r['prompt']:,} | {r['completion']:,} | "
                 f"{r['reasoning']:,} | {r['truncated']} |")
    txt = "\n".join(L) + "\n"
    if not args.llm:
        open(os.path.join(QA_DIR, "summary.md"), "w").write(txt)
    print(txt)


def per_question(path):
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            p = extract(r["raw"])
            out[r["id"]] = (f1_locomo(p, r["answer"], r["category"]), f1_simple(p, r["answer"]))
    return out


def stage_sig(args):
    """Same generator, same questions: per-question F1 of each system against --base, two-sided paired
    randomization test (scripts/baselines/sig_test.py, 100k sign flips) with a 95% bootstrap CI of the difference."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
    from sig_test import paired, stars
    L = [f"paired per-question test vs {args.base} (same LLM, same 1531 questions); Δ in F1 points", "",
         "| LLM | system | Δ Overall F1 (LoCoMo eval) | 95% CI | p | | Δ (Mem0 eval) | 95% CI | p | |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(glob.glob(os.path.join(QA_DIR, "*"))):
        files = {os.path.basename(f)[:-6]: f for f in glob.glob(os.path.join(d, "*.jsonl"))}
        if args.base not in files:
            continue
        B = per_question(files[args.base])
        for name in sorted(files):
            if name == args.base:
                continue
            A = per_question(files[name])
            ids = sorted(set(A) & set(B))
            if len(ids) < len(B):
                continue                                         # still running
            cells = []
            for j in (0, 1):
                r = paired([A[i][j] for i in ids], [B[i][j] for i in ids])
                cells += [f"{r['delta']:+.2f}", f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]", f"{r['p_rand']:.2g}",
                          stars(r["p_rand"]) or "n.s."]
            L.append(f"| {os.path.basename(d)} | {name} | " + " | ".join(cells) + " |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(QA_DIR, f"sig_vs_{args.base}.md"), "w").write(txt)
    print(txt)


def main():
    global DATA, RAW, QA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report", "sig"])
    ap.add_argument("--base", default="qrranker", help="sig: the system everything is paired against")
    ap.add_argument("--llm", default="", help="run: the model; report: optional filter (comma list)")
    ap.add_argument("--system", action="append", default=[], help="NAME=SPEC, repeatable")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-out", type=int, default=0, help=
                    "0: 1024 for gpt-4o / gpt-4.1 / gpt-3.5 models, else 16384 (reasoning tokens count)")
    ap.add_argument("--data", default=DATA, help="the LoCoMo candidate file")
    ap.add_argument("--raw", default=RAW, help="the raw LoCoMo-10 file (locomo10.json), for the gold answers")
    ap.add_argument("--qa-dir", default=QA_DIR, help="replies <qa-dir>/<llm>/<NAME>.jsonl and the reports")
    args = ap.parse_args()
    DATA, RAW, QA_DIR = args.data, args.raw, args.qa_dir
    if args.stage == "run":
        if not args.max_out:
            args.max_out = 1024 if args.llm.startswith(("gpt-4o", "gpt-4.1", "gpt-3.5")) else 16384
        stage_run(args)
    elif args.stage == "sig":
        stage_sig(args)
    else:
        args.llm = [x for x in args.llm.split(",") if x]
        stage_report(args)


if __name__ == "__main__":
    main()
