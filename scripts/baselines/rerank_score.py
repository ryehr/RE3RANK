"""Score the nine-set candidates with an external reranker baseline.

The first stage is identical to every RE3Rank run: the top-50 candidates that already sit in
the candidate files (same embedding retriever), so only the reranking step differs.

  --method qwen3      Qwen3-Reranker pointwise cross-encoder. P(yes) against P(no) at
                      the last position, following the model card's vLLM recipe exactly
                      (chat template without thinking, the fixed suffix, max_tokens 1,
                      allowed_token_ids = [yes, no]).
  --method jina       jina-reranker-v3.5 listwise ("last-but-not-late"): one forward
                      over the query and the whole candidate list, cosine between the
                      query embedding and each document embedding. Its own rerank()
                      handles blocking, so the call is just rerank(query, documents).
                      Needs trust_remote_code (the repo ships modeling.py; it imports
                      only numpy/torch/transformers.models.qwen3).
  --method zerank     zerank-2 pointwise cross-encoder through sentence-transformers,
                      as its card documents: predict() returns the raw "Yes" logit per
                      pair, which is what we rank by (the (x/5).sigmoid() in the card is
                      cosmetic and order-preserving).
  --method grouprank  Diver-GroupRank groupwise. The card's system/user prompts verbatim
                      and the protocol from the official Reranker/rerank_groupwise.py:
                      groups of 10, every candidate covered once per epoch, 2 epochs over
                      independently shuffled orders, documents cut to 3000 chars with
                      newlines collapsed, integer 0-10 scores divided by 10 and averaged
                      across the epochs a document appears in, a group whose JSON fails
                      to parse scores 0 for all of its documents.
  --method rearank    REARANK (le723z/Rearank-7B) listwise reasoning reranker, the official protocol
                      (rearank_official.py, ported from lezhang7/Rearank): sliding window 20 /
                      step 10 from the bottom of the 50 candidates (4 windows, each on the order the
                      previous one left), passages cut to 400 characters, the model's chat template,
                      greedy, max_tokens 2048, the official parser; score = 50 - final position. Every
                      window's reply goes to <out dir>/raw/<set>.s<shard>.jsonl.
  --method reasonrank ReasonRank-7B / -32B listwise reasoning reranker, the official protocol of
                      run_rank_llm.sh (reasonrank_official.py, ported from 8421BCD/reasonrank):
                      prompt_mode rank_GPT_reasoning, passages cut to --doc-maxlen tokens (512, the
                      official value for BRIGHT / R2MED / its training data; our sets have no official
                      entry), sliding window 20 / step 10 over the 50 candidates (4 windows, one after
                      another), greedy, max_tokens 3072 + 100, the official parser; score = 50 - final
                      position; raw replies as for rearank.

Each shard writes <out>.s<shard>.json, {item id: [score per candidate, in the candidate file's
order]} (higher = more relevant); --merge joins the shards into <out>. Driven by run_baseline.sh:

  CUDA_VISIBLE_DEVICES=0 python scripts/baselines/rerank_score.py --data locomo --method qwen3 \\
      --model Qwen/Qwen3-Reranker-4B --out outputs/baselines/qwen3-reranker-4b/locomo.json --shard 0 --nshards 8
  python scripts/baselines/rerank_score.py --data locomo --method qwen3 --model Qwen/Qwen3-Reranker-4B \\
      --out outputs/baselines/qwen3-reranker-4b/locomo.json --nshards 8 --merge
"""
import argparse
import json
import math
import os
import random
import re

from re3rank import paths

QWEN_SYS = ('Judge whether the Document meets the requirements based on the Query and '
            'the Instruct provided. Note that the answer can only be "yes" or "no".')
QWEN_TASK = 'Given a web search query, retrieve relevant passages that answer the query'
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_PREFIX = f"<|im_start|>system\n{QWEN_SYS}<|im_end|>\n<|im_start|>user\n"

GR_SYS = '''Your task is to evaluate and rank documents based on how well they help answer the given query. Follow this evaluation priority:
1. PRIMARY: Usefulness & Helpfulness - Does the document provide actionable information, solutions, or direct answers that help address the user's needs?
2. SECONDARY: Relevance - Does the document contain information related to the query topic?

Evaluation Process:
1. First, identify the user's core intent and what kind of help they need from the query
2. For each document, assess:
   - How directly it addresses the user's intent
   - What actionable information or answers it provides
   - How much it helps solve the user's problem or need
3. Compare documents against each other to ensure proper ranking
4. Assign scores that reflect the relative usefulness ranking

Scoring Scale (0-10):
- 9-10: Extremely helpful, directly answers the query with actionable information
- 7-8: Very helpful, provides substantial useful information for the query
- 5-6: Moderately helpful, contains some useful information but incomplete
- 3-4: Minimally helpful, limited useful information despite topic relevance
- 1-2: Barely helpful, mentions related topics but provides little useful information
- 0: Not helpful at all, cannot assist with answering the query
'''

GR_USER = '''I will provide you {TOPK} documents, each indicated by a numerical identifier []. Score these documents based on their Usefulness and Relevance to the query.
Query:
{QUERY}

Documents:
{PASSAGES}

## Final Output Format
You must structure your response in exactly two parts: provide your brief reasoning process first, then output final scores in JSON format like below, with document IDs as string keys and integer scores as values for all {TOPK} documents. 
The reasoning process and answer are enclosed within <reason> </reason> and <answer> </answer> tags, respectively. Do NOT output anything outside the specified tags. Follow this exact format:
<reason> 
[Analyze each document's usefulness and relevance to the query, explaining your scoring rationale]
</reason>
<answer> 
```json
{{"[1]": 5, "[2]": 3, "[3]": 8}}
``` 
</answer>
'''


def doc_text(p):
    """Candidate text for the external rerankers: title, a newline, then the paragraph (the paragraph
    alone when the title is empty)."""
    t = (p.get("title") or "").strip()
    b = p["paragraph_text"]
    return f"{t}\n{b}" if t else b


def extract_scores(out):
    """The official parser from Reranker/rerank_groupwise.py, failure raises."""
    if "<answer>" in out:
        seg = out.split("<answer>")[-1]
    else:
        seg = out
    js = re.findall(r"\{[\s\S]*?\}", seg)
    if not js:
        js = re.findall(r"\{[\s\S]*?\}", out)
    s = js[-1].strip()
    s = re.sub(r"(\[\d+\])\]", r"\1", s)
    if s.count("{") > s.count("}"):
        s += "}"
    s = re.sub(r"\s+", " ", s).replace("\n", " ").strip()
    return dict(json.loads(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="candidate file, or the name of an evaluation set")
    ap.add_argument("--method", choices=["qwen3", "grouprank", "jina", "zerank", "rearank", "reasonrank"],
                    required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="merge <out>.s*.json and exit")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--instruct", default=QWEN_TASK)
    ap.add_argument("--qwen-max-len", type=int, default=8192)
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--num-epoch", type=int, default=2)
    ap.add_argument("--max-doc-chars", type=int, default=3000)
    ap.add_argument("--gr-max-len", type=int, default=16384)
    ap.add_argument("--st-batch", type=int, default=32)
    ap.add_argument("--st-max-len", type=int, default=8192)
    ap.add_argument("--gr-max-tokens", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--doc-maxlen", type=int, default=512, help="reasonrank: passage length in tokens")
    args = ap.parse_args()

    if args.merge:
        out = {}
        for i in range(args.nshards):
            f = f"{args.out}.s{i}.json"
            if os.path.exists(f):
                out.update(json.load(open(f)))
        json.dump(out, open(args.out, "w"))
        print(f"merged {args.nshards} shards -> {args.out}: {len(out)} items", flush=True)
        return

    data = json.load(open(paths.eval_file(args.data) if args.data in paths.EVAL_SETS else args.data))
    if args.limit:
        data = data[: args.limit]
    mine = [(i, it) for i, it in enumerate(data) if i % args.nshards == args.shard]
    print(f"[shard {args.shard}/{args.nshards}] {len(mine)} items", flush=True)

    scores = {}

    if args.method == "jina":
        import torch
        from transformers import AutoModel
        try:
            m = AutoModel.from_pretrained(args.model, dtype="auto", trust_remote_code=True)
        except TypeError:                      # transformers < 4.56 spells it torch_dtype
            m = AutoModel.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True)
        m = m.cuda().eval()
        for c, (idx, it) in enumerate(mine):
            docs = [doc_text(p) for p in it["paragraphs"]]
            with torch.no_grad():
                res = m.rerank(it["question"], docs)
            v = [0.0] * len(docs)
            for r in res:
                v[r["index"]] = float(r["relevance_score"])
            scores[str(it["id"])] = v
            if (c + 1) % 200 == 0:
                print(f"[shard {args.shard}] {c+1}/{len(mine)}", flush=True)
        json.dump(scores, open(f"{args.out}.s{args.shard}.json", "w"))
        print(f"[shard {args.shard}] wrote {len(scores)} items", flush=True)
        return

    if args.method == "zerank":
        import torch
        from sentence_transformers import CrossEncoder
        m = CrossEncoder(args.model, model_kwargs={"torch_dtype": torch.bfloat16},
                         max_length=args.st_max_len, device="cuda")
        pairs, owner = [], []
        for idx, it in mine:
            for j, p in enumerate(it["paragraphs"]):
                pairs.append((it["question"], doc_text(p)))
                owner.append((str(it["id"]), j, len(it["paragraphs"])))
        print(f"[shard {args.shard}] {len(pairs)} pairs", flush=True)
        sc = m.predict(pairs, batch_size=args.st_batch, convert_to_numpy=True,
                       show_progress_bar=True)
        for v, (qid, j, n) in zip(sc, owner):
            scores.setdefault(qid, [0.0] * n)[j] = float(v)
        json.dump(scores, open(f"{args.out}.s{args.shard}.json", "w"))
        print(f"[shard {args.shard}] wrote {len(scores)} items", flush=True)
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt          # vllm 0.28: no longer vllm.inputs.data
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.method == "rearank":
        import rearank_official as RO
        # agent.QwenClient: LLM(model) with the model's own context length, explicit sampling params
        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem)
        sp = SamplingParams(**RO.SAMPLING)
        items = {str(it["id"]): RO.to_item(it) for idx, it in mine}
        pos = {q: RO.window_positions(len(v["hits"])) for q, v in items.items()}
        raw = {q: [] for q in items}
        for w in range(max(len(v) for v in pos.values())):
            batch = [(q, *pos[q][w]) for q in items if w < len(pos[q])]
            texts = [tok.apply_chat_template(RO.create_permutation_instruction_rearank(items[q], a, b),
                                             tokenize=False, add_generation_prompt=True) for q, a, b in batch]
            outs = llm.generate(texts, sp, use_tqdm=True)
            for (q, a, b), o in zip(batch, outs):
                reply = o.outputs[0].text
                items[q] = RO.receive_permutation(items[q], reply, rank_start=a, rank_end=b)
                raw[q].append({"window": [a, b], "out_tok": len(o.outputs[0].token_ids),
                               "finish": o.outputs[0].finish_reason, "answer": RO.answer_parsed(reply),
                               "prompt_tok": len(o.prompt_token_ids), "reply": reply})
            print(f"[shard {args.shard}] window {w + 1}: {len(batch)} prompts", flush=True)
        for q, v in items.items():
            scores[q] = RO.final_scores(v)
        rd = os.path.join(os.path.dirname(os.path.abspath(args.out)), "raw")
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, f"{os.path.basename(args.out)[:-5]}.s{args.shard}.jsonl"), "w") as f:
            for q, r in raw.items():
                f.write(json.dumps({"id": q, "windows": r}) + "\n")
        W = [x for r in raw.values() for x in r]
        print(f"[shard {args.shard}] windows {len(W)}: no closed <answer> {sum(not x['answer'] for x in W)}, "
              f"cut at max_tokens {sum(x['finish'] == 'length' for x in W)}, "
              f"mean reply tokens {sum(x['out_tok'] for x in W) / max(1, len(W)):.0f}", flush=True)
    elif args.method == "reasonrank":
        import reasonrank_official as RR
        # RankListwiseOSLLM(vllm_batched=True): LLM(model) with its own context length; the prompts are built
        # with the engine's tokenizer, as the official create_prompt does
        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem)
        sp = SamplingParams(**RR.SAMPLING)
        P = RR.ReasonRankPrompter(llm.get_tokenizer(), args.doc_maxlen)
        results = {str(it["id"]): RR.to_result(it) for idx, it in mine}
        pos = {q: RR.window_positions(len(r.candidates)) for q, r in results.items()}
        raw = {q: [] for q in results}
        for w in range(max(len(v) for v in pos.values())):
            batch = [(q, *pos[q][w]) for q in results if w < len(pos[q])]
            texts = [P.create_prompt(results[q], a, b)[0] for q, a, b in batch]
            outs = llm.generate(texts, sp, use_tqdm=True)
            for (q, a, b), o in zip(batch, outs):
                reply = o.outputs[0].text
                results[q] = P.receive_permutation(results[q], reply, a, b)
                raw[q].append({"window": [a, b], "out_tok": len(o.outputs[0].token_ids),
                               "finish": o.outputs[0].finish_reason, "answer": RR.answer_parsed(reply),
                               "prompt_tok": len(o.prompt_token_ids), "reply": reply})
            print(f"[shard {args.shard}] window {w + 1}: {len(batch)} prompts", flush=True)
        for q, r in results.items():
            scores[q] = RR.final_scores(r)
        rd = os.path.join(os.path.dirname(os.path.abspath(args.out)), "raw")
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, f"{os.path.basename(args.out)[:-5]}.s{args.shard}.jsonl"), "w") as f:
            for q, r in raw.items():
                f.write(json.dumps({"id": q, "windows": r}) + "\n")
        W = [x for r in raw.values() for x in r]
        print(f"[shard {args.shard}] windows {len(W)}: official pattern missed {sum(not x['answer'] for x in W)}, "
              f"cut at max_tokens {sum(x['finish'] == 'length' for x in W)}, "
              f"mean reply tokens {sum(x['out_tok'] for x in W) / max(1, len(W)):.0f}", flush=True)
    elif args.method == "qwen3":
        llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.qwen_max_len,
                  enable_prefix_caching=True, gpu_memory_utilization=args.gpu_mem)
        suf = tok.encode(QWEN_SUFFIX, add_special_tokens=False)
        t_yes = tok("yes", add_special_tokens=False).input_ids[0]
        t_no = tok("no", add_special_tokens=False).input_ids[0]
        sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20,
                            allowed_token_ids=[t_yes, t_no])
        pre = tok.encode(QWEN_PREFIX, add_special_tokens=False)
        budget = args.qwen_max_len - len(pre) - len(suf)
        prompts, owner = [], []
        for idx, it in mine:
            for j, p in enumerate(it["paragraphs"]):
                # built by hand, not through apply_chat_template: this repo ships a
                # chat_template.jinja that transformers 5.x prefers and that reads the
                # query and document from custom "query"/"document" roles -- feeding it
                # the card's old system+user messages renders BOTH fields empty, so every
                # candidate would get an identical prompt and an identical score, with no
                # error anywhere. This literal is byte-identical to what that template
                # emits when driven correctly (verified), and to the card's recipe.
                body = (f"<Instruct>: {args.instruct}\n<Query>: {it['question']}\n"
                        f"<Document>: {doc_text(p)}")
                ids = pre + tok(body, add_special_tokens=False)["input_ids"][:budget] + suf
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                owner.append((str(it["id"]), j, len(it["paragraphs"])))
        print(f"[shard {args.shard}] {len(prompts)} pairs", flush=True)
        outs = llm.generate(prompts, sp, use_tqdm=True)
        for o, (qid, j, n) in zip(outs, owner):
            lp = o.outputs[0].logprobs[-1]
            ty = lp[t_yes].logprob if t_yes in lp else -10.0
            tn = lp[t_no].logprob if t_no in lp else -10.0
            ey, en = math.exp(ty), math.exp(tn)
            scores.setdefault(qid, [0.0] * n)[j] = ey / (ey + en)
    else:
        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
                  max_model_len=args.gr_max_len, gpu_memory_utilization=args.gpu_mem)
        sp = SamplingParams(temperature=0.3, top_p=0.8, max_tokens=args.gr_max_tokens,
                            seed=args.seed)
        prompts, owner = [], []
        for idx, it in mine:
            n = len(it["paragraphs"])
            for ep in range(args.num_epoch):
                order = list(range(n))
                # per item and epoch, independent of how the set is sharded
                random.Random(args.seed + 7919 * ep + idx).shuffle(order)
                for a in range(0, n, args.group_size):
                    grp = order[a: a + args.group_size]
                    txt = "".join(
                        "[{}]. {}\n\n".format(k + 1,
                                              re.sub(r"\n+", " ", doc_text(it["paragraphs"][g]))[: args.max_doc_chars])
                        for k, g in enumerate(grp))
                    user = GR_USER.format(QUERY=it["question"], PASSAGES=txt, TOPK=len(grp))
                    prompts.append(tok.apply_chat_template(
                        [{"role": "system", "content": GR_SYS}, {"role": "user", "content": user}],
                        tokenize=False, add_generation_prompt=True))
                    owner.append((str(it["id"]), n, grp))
        print(f"[shard {args.shard}] {len(prompts)} groups", flush=True)
        outs = llm.generate(prompts, sp, use_tqdm=True)
        acc, bad = {}, 0
        for o, (qid, n, grp) in zip(outs, owner):
            acc.setdefault(qid, [[] for _ in range(n)])
            try:
                sc = extract_scores(o.outputs[0].text)
                got = {}
                for k, v in sc.items():
                    k = int(str(k).strip().strip("[]"))
                    if 1 <= k <= len(grp):
                        got[grp[k - 1]] = float(v) / 10.0
            except Exception:
                bad += 1
                got = {}
            for g in grp:                      # unparsed / missing ids score 0
                acc[qid][g].append(got.get(g, 0.0))
        for qid, per in acc.items():
            scores[qid] = [sum(v) / len(v) if v else 0.0 for v in per]
        print(f"[shard {args.shard}] groups that failed to parse: {bad}/{len(owner)}", flush=True)

    json.dump(scores, open(f"{args.out}.s{args.shard}.json", "w"))
    print(f"[shard {args.shard}] wrote {len(scores)} items", flush=True)


if __name__ == "__main__":
    main()
