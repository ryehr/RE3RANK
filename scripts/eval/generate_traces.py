"""Generate the reasoning trace of every item of a candidate file with vLLM.

The generator sees the plain system prompt and the user turn (candidates + question) in the model's
native thinking template, with the thinking block forced open; decoding is greedy (T=0) and stops
at the closing tag (at most --max-new tokens). The output is the RAW open trace, byte-identical to
what was generated: {id: trace}. Items whose prompt does not fit the context window, and traces
shorter than 20 characters, are left out (the scorer then falls back to the question).

  CUDA_VISIBLE_DEVICES=0 python scripts/eval/generate_traces.py --model outputs/ckpt/re3rank-4b/final \
      --tokenizer Qwen/Qwen3-4B --data musique --out traces/musique.s0.json --shard 0 --nshards 8
  python scripts/eval/generate_traces.py --merge traces/musique.s*.json --out traces/musique.json
"""
import argparse
import json
import os

from re3rank import backbone as bbk
from re3rank import paths
from re3rank.prompts import RR_SYSTEM, user_msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="candidate file, or the name of an evaluation set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="", help="generator weights (a checkpoint dir or a hub id)")
    ap.add_argument("--tokenizer", default="", help="tokenizer (default: --model); use the base model id")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--merge", nargs="+", default=[], help="merge shard files into --out")
    args = ap.parse_args()

    if args.merge:
        out = {}
        for f in args.merge:
            out.update(json.load(open(f)))
        json.dump(out, open(args.out, "w"), ensure_ascii=False)
        print(f"merged {len(args.merge)} shards -> {args.out}: {len(out)} items")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok_id = args.tokenizer or args.model
    FMT = bbk.fmt_for(tok_id)
    tok = AutoTokenizer.from_pretrained(tok_id)
    path = paths.eval_file(args.data) if args.data in paths.EVAL_SETS else args.data
    data = json.load(open(path))
    if args.limit:
        data = data[: args.limit]
    data = data[args.shard::args.nshards]
    prompts, keep = [], []
    for x in data:
        msg = user_msg(x["question"], x["paragraphs"])
        if len(tok(msg)["input_ids"]) + args.max_new + 400 > args.max_model_len:
            continue                      # would not fit the context window
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": RR_SYSTEM}, {"role": "user", "content": msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True) + FMT.think_open)
        keep.append(x)
    print(f"shard {args.shard}/{args.nshards}: {len(keep)} of {len(data)} items", flush=True)

    llm_kw = {}
    if FMT.fam == "gemma4":   # multimodal wrapper, text-only use
        llm_kw["limit_mm_per_prompt"] = {"image": 0, "audio": 0, "video": 0}
    llm = LLM(model=args.model, tokenizer=tok_id, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, disable_log_stats=True, **llm_kw)
    # the reasoning markers are special tokens on Gemma: keep them in the text so the split below
    # sees them; stop on the closing id as well as on its text
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new, stop=[FMT.think_close],
                        stop_token_ids=[FMT.think_close_id(tok)], skip_special_tokens=(FMT.fam == "qwen"))
    outs = llm.generate(prompts, sp)
    traces, bad = {}, 0
    for x, o in zip(keep, outs):
        t = o.outputs[0].text
        # everything after the opener (the stop cuts before the closing tag; no tag = truncated run)
        tr = t.split(FMT.think_open, 1)[1] if FMT.think_open in t else t
        if FMT.fam != "qwen":
            tr = tr.split(FMT.think_close, 1)[0]
        if len(tr.strip()) >= 20:
            traces[str(x["id"])] = tr
        else:
            bad += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(traces, open(args.out, "w"), ensure_ascii=False)
    print(f"wrote {len(traces)} ({bad} unusable) -> {args.out}")


if __name__ == "__main__":
    main()
