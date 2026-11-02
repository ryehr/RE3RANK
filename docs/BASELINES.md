# Baselines and LoCoMo downstream QA

This page covers the external rerankers RE3Rank is compared with (`scripts/baselines/`) and the LoCoMo
downstream question answering (`scripts/downstream/`).

Every baseline reranks **the same first stage** as RE3Rank: the top-50 candidates (Qwen3-Embedding-8B)
stored in the candidate files of the nine evaluation sets (`re3rank.paths.eval_file(name)`, built by
`scripts/data/`). Only the reranking step differs, and every system is scored with the same metrics:

* **conditional R@k** = |gold in top-k| / |gold in the 50 candidates|, over the queries with at least one gold
  candidate in the pool;
* **NDCG@10** with binary relevance and the ideal DCG over min(10, |gold in pool|);
* ties keep the first-stage order (this matters for GroupRank, whose scores are averaged integers);
* for HotpotQA the scripts also print the absolute recall with 2 golds per question (`--gold-denom 2`);
* the macro average is over the nine sets.

Outputs go under `$RE3RANK_OUT` (default `outputs/`), candidate files are read from `$RE3RANK_DATA`
(default `data/`), and model weights come from the Hugging Face cache (`$HF_HOME`). All commands are run
from the repository root with the package importable (`pip install -e .`; the shell drivers also put the
repository root on `PYTHONPATH`).

## Environments

| environment | used by | packages (versions used for the paper) |
|---|---|---|
| main | every script except the QRRanker scorer | torch 2.13, transformers 5.16.1, vLLM 0.28.0, sentence-transformers 6.0.1, ftfy 6.3.1 (ReasonRank), numpy, scipy; for the QA: openai, tiktoken, nltk |
| QRRanker | `qrranker_score.py` (`run_qrranker.sh`, `rank_locomo.sh SYSTEM=qrranker`) | transformers 4.53.0, huggingface_hub 0.36.2, kernels 0.9.0; torch and the rest from the main environment |

The released QRRanker checkpoint ships custom modeling code for the transformers ~4.53 cache API
(`DynamicCacheWithQuery` needs `_seen_tokens`, dropped in 4.54; the rope registry and config field names
moved again in 5.x), so it gets its own small environment layered on the main one. A newer `kernels`
package makes transformers 4.53 fail to import, hence the pin:

```bash
python -m venv --system-site-packages venv-qrranker        # with the main environment's python
venv-qrranker/bin/pip install transformers==4.53.0 huggingface_hub==0.36.2 kernels==0.9.0
export QRRANKER_PY=venv-qrranker/bin/python
```

## Models

`bash scripts/baselines/download.sh` downloads all of them into the Hugging Face cache. The revisions are
the ones used for the paper.

| baseline | model id | revision | `--method` | type |
|---|---|---|---|---|
| QRRanker (4B) | `MindscapeRAG/QRRanker` | `8534b94` | (`qrranker_score.py`) | attention heads (16 QR heads) |
| Qwen3-Reranker (0.6B / 4B / 8B) | `Qwen/Qwen3-Reranker-0.6B` / `-4B` / `-8B` | `e61197e` / `22e6836` / `77d193c` | `qwen3` | pointwise |
| zerank-2 (4B) | `zeroentropy/zerank-2-reranker` | `5eae30d` | `zerank` | pointwise |
| jina-reranker-v3.5 (0.6B) | `jinaai/jina-reranker-v3.5` | `e8a93f3` | `jina` | listwise embedding |
| GroupRank (7B / 32B) | `AQ-MedAI/Diver-GroupRank-7B` / `-32B` | `42a9504` / `820775a` | `grouprank` | groupwise generative |
| REARANK (7B) | `le723z/Rearank-7B` | `530a7e1` | `rearank` | listwise reasoning |
| ReasonRank (7B / 32B) | `liuwenhan/reasonrank-7B` / `-32B` | `3444046` / `bc8629c` | `reasonrank` | listwise reasoning |

Check each model card for its license and terms of use.

## Protocols

Candidate text for the external rerankers is `title + "\n" + paragraph` (the paragraph alone when the
title is empty), except where a baseline's official code builds its own passage string (QRRanker,
REARANK, ReasonRank).

**First stage** (no reranking). The candidate file's own order: `eval_scores.py --first-stage`.

**QRRanker** (`qrranker_score.py`). The released checkpoint (16 trained QR heads on a 25-layer truncated
Qwen3-4B-Instruct-2507, bf16) driven with the RE3Rank prompt (`re3rank.prompts.build_prompt`: the user
turn lists the 50 candidates as `[k] title: text`, the `title: ` prefix dropped when the title is empty,
then the question), the question tokens as query rows, no truncation (`max_length` 262144). The score of a
candidate is the attention mass of the question rows on its tokens (averaged over the rows, recomputed
from the cached query and key states) summed over the 16 heads of the checkpoint's config, following the
code of the model card; ties in the ranking follow `torch.argsort`. This makes the row comparable with the RE3Rank scorer rather than with the
QRRanker paper's own protocol.

**Qwen3-Reranker** (`--method qwen3`). The model card's vLLM recipe: the fixed system prompt, the user turn
`<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: ...\n<Document>: ...`,
the fixed assistant suffix with an empty thinking block, built token by token rather than through the
chat template (the repository's `chat_template.jinja`, which transformers 5.x prefers, reads the query and
document from custom roles and would otherwise render both empty). The user-turn tokens are cut so that
the prompt fits `--qwen-max-len` 8192. One generated token with `allowed_token_ids = [yes, no]` and the
top-20 logprobs; score = P(yes) / (P(yes) + P(no)), a token absent from the top-20 counting as logprob -10.

**zerank-2** (`--method zerank`). `sentence_transformers.CrossEncoder` as on the model card, bf16,
`max_length` 8192, batch 32; the score is the raw "Yes" logit returned by `predict()` (the card's
`sigmoid(x / 5)` is order-preserving and skipped).

**jina-reranker-v3.5** (`--method jina`). The model's own `rerank(query, documents)` (trust_remote_code; one
listwise forward over the query and all 50 candidates, blocking handled inside), score = its
`relevance_score`.

**GroupRank** (`--method grouprank`). The model card's system and user prompts verbatim and the protocol of
the official `rerank_groupwise.py`: groups of 10, every candidate covered once per epoch, 2 epochs over
independently shuffled orders (seeded per item and epoch: seed 666 + 7919 * epoch + item index), each
document's newlines collapsed and cut to 3000 characters; vLLM with temperature 0.3, top-p 0.8, at most
8000 new tokens, sampling seed 666, `max_model_len` 16384; the integer 0-10 scores are divided by 10 and
averaged over the epochs, and a group whose JSON answer fails the official parser scores 0 for all its
documents.

**REARANK** (`--method rearank`, `rearank_official.py`, from lezhang7/Rearank). Passages as the official
`run_retriever` builds them (`Title: <title> Content: <text>`, whitespace collapsed; the empty
`Title: Content: ` dropped) and cut to their first **400 characters** by the official prompt builder;
sliding window 20, step 10, from the bottom of the 50 candidates to the top: (30, 50), (20, 40), (10, 30),
(0, 20), each window on the order the previous one left; the model's chat template; greedy (temperature 0,
top-p 1, repetition penalty 1), at most 2048 new tokens; the official parser (ids inside `<answer>`,
duplicates dropped, missing ids appended in their current order). Score = 50 - final position.

**ReasonRank** (`--method reasonrank`, `reasonrank_official.py`, from 8421BCD/reasonrank). The settings of
the official `run_rank_llm.sh`: prompt mode `rank_GPT_reasoning` (the reasoning system prompt and one
user message: prefix, `[i] <passage>` lines, post prompt; the model's chat template; `ftfy.fix_text` over
the prompt); passages `Title: <title> Content: <text>` (title only when non-empty), cut to
`--doc-maxlen` **512 tokens** of the model's tokenizer, `[n]` rewritten to `(n)`. The official passage
length is set per dataset (100 for TREC/BEIR, 512 for BRIGHT, R2MED and its training data) and has no
entry for these sets; 512 is used. Sliding window 20 / step 10 over the 50 candidates (the same four
windows), greedy, at most 3072 + 100 new tokens, the official parser. Score = 50 - final position.

The listwise reasoning rerankers also write every window's reply (with its prompt / reply token counts
and finish reason) to `<out dir>/raw/<set>.s<shard>.jsonl`.

## Running the baselines

```bash
bash scripts/baselines/download.sh

# one reranker on the nine sets (one shard per GPU, then merged and evaluated)
TAG=qwen3-reranker-4b MODEL=Qwen/Qwen3-Reranker-4B METHOD=qwen3 bash scripts/baselines/run_baseline.sh
TAG=rearank-7b MODEL=le723z/Rearank-7B METHOD=rearank bash scripts/baselines/run_baseline.sh
#   smoke test: LIMIT=40 SETS="locomo"; GPUs: NGPU (default: all), vLLM memory share: GPU_MEM (0.85)

# the released QRRanker (its own environment, see above)
QRRANKER_PY=venv-qrranker/bin/python bash scripts/baselines/run_qrranker.sh

# everything: the first-stage row, the ten external rerankers, QRRanker, and the macro-9 table
QRRANKER_PY=venv-qrranker/bin/python bash scripts/baselines/run_all.sh
```

Per set, `$RE3RANK_OUT/baselines/<tag>/` holds `<set>.json` (scores, `{item id: [score per candidate]}`),
`<set>.txt` (metrics) and `logs/`; QRRanker keeps its per-item shard records `<set>.s<gpu>.json` instead
of scores. A set that is already scored is only re-evaluated. Single steps:

```bash
# metrics of a score file (or of the first-stage order)
python scripts/baselines/eval_scores.py --data musique --scores outputs/baselines/zerank-2/musique.json
python scripts/baselines/eval_scores.py --data hotpotqa --first-stage --gold-denom 2

# macro-9 table of R@3 / NDCG@10; RE3Rank runs (scripts/eval/run_eval.sh) mix in by their directory
python scripts/baselines/summary.py first-stage=First-stage qrranker=QRRanker \
    jina-reranker-v3.5="jina-reranker-v3.5 (0.6B)" outputs/eval/re3rank-4b_final/scores=RE3Rank-4B

# paired per-query significance test of two rerankers on one set
python scripts/baselines/sig_test.py musique dump:outputs/eval/re3rank-4b_final/scores \
    'official:outputs/baselines/qrranker/{st}.s*.json'
```

`sig_test.py` sources: `scores:<dir>` (a baseline directory), `official:<glob>` (QRRanker shard records,
`{st}` = the set) and `dump:<dir>` (per-item records of `scripts/eval/score.py --out`, written without
`--keep-nogold`). It reports the two-sided paired randomization (sign-flip) test with 100,000 flips, a
paired t-test and a 95% bootstrap CI (20,000 resamples) of the mean difference, over the queries with a
gold candidate in the pool.

vLLM generation is not bit-reproducible across hardware and versions, so reruns of the generative
baselines can differ slightly from the paper.

## LoCoMo downstream QA

The protocol of the QRRanker paper's Table 3 (`scripts/downstream/locomo_qa.py`):

* **Questions**: the 1531 questions of the LoCoMo candidate file, i.e. the 1540 non-adversarial
  (category != 5) LoCoMo-10 questions minus the 9 whose evidence turns do not resolve to a chunk. The 29
  questions without a gold chunk among the 50 candidates are asked too. Gold answers are read back from
  the raw LoCoMo-10 release file (`--raw`, default `$RE3RANK_DATA/raw/locomo/locomo10.json`) through the
  item id (sample id + index in the sample's non-adversarial QA list), with the question text checked.
* **Context**: each system reranks the same 50 first-stage chunks; the **top-3** chunk texts (each chunk
  starts with its session date), in ranked order, joined by blank lines.
* **Prompt**: the paper's LoCoMo QA prompt (its App. J.4, Mem0's answer prompt) with the context and the
  question filled in, sent as one system message, the way Mem0's evaluation does.
* **Models**: OpenAI chat models; the paper uses `gpt-4o-mini`, `gpt-5-mini` and `gpt-6-luna`.
  `gpt-4o*` / `gpt-4.1*` / `gpt-3.5*` models run at temperature 0 with at most 1024 completion tokens;
  every other model is treated as a reasoning model: default temperature and reasoning effort, at most
  16384 completion tokens (reasoning included). Failed calls are retried with exponential backoff (8
  attempts).
* **Answer**: the text after the last `answer:` marker if the reply has one, else the whole reply, with
  surrounding `*` stripped.
* **F1**, per category (single-hop = 4, multi-hop = 1, temporal = 2, open-domain = 3; 841 / 281 / 320 / 89
  questions) and Overall = the mean over all questions:
  * *LoCoMo evaluation* (main): lower-case, strip punctuation and commas, drop `a|an|the|and`, Porter
    stems, token-multiset F1; for multi-hop, gold and prediction are split on commas and the F1 is the mean
    over the gold parts of the best-matching prediction part; for open-domain, the gold is cut at the
    first `;`;
  * *Mem0 evaluation*: lower-case, `. , ! ?` to spaces, set-of-tokens F1.
* **Significance**: per LLM, every system against `--base` on the same questions: two-sided paired
  randomization test of the per-question F1 (100,000 sign flips) and a 95% bootstrap CI of the difference.

### Rankings

Every question needs a ranking, including those without a gold chunk in the pool:

```bash
# QRRanker: all 1531 orders (--order-out)
SYSTEM=qrranker QRRANKER_PY=venv-qrranker/bin/python bash scripts/downstream/rank_locomo.sh
#   -> qrranker=order:outputs/downstream/locomo_rank/qrranker.order.json

# RE3Rank: score.py --keep-nogold --out on the checkpoint's LoCoMo traces (from scripts/eval/run_eval.sh)
SYSTEM=re3rank NAME=re3rank-4b CKPT=outputs/ckpt/re3rank-4b/final BASE=Qwen/Qwen3-4B HEADS=23-10,21-8 \
    bash scripts/downstream/rank_locomo.sh
#   -> re3rank-4b=scores:outputs/downstream/locomo_rank/re3rank-4b.json
```

An external reranker needs no extra step: its score file `outputs/baselines/<tag>/locomo.json` covers every
question (`zerank-2=scores:outputs/baselines/zerank-2/locomo.json`). The first-stage order is `first=first`.

### QA runs, tables and test

```bash
export OPENAI_API_KEY=...        # read from the environment only
LLMS="gpt-4o-mini gpt-5-mini gpt-6-luna" bash scripts/downstream/run_locomo_qa.sh first=first \
    qrranker=order:outputs/downstream/locomo_rank/qrranker.order.json \
    re3rank-4b=scores:outputs/downstream/locomo_rank/re3rank-4b.json
```

This runs one process per LLM (12 parallel requests, `WORKERS`), caches every reply in
`$RE3RANK_OUT/downstream/locomo_qa/<llm>/<name>.jsonl` (with the top-3 chunk ids, whether a gold chunk is
among them, and the token usage; an interrupted run resumes), then writes `summary.md` (both F1 tables
and the usage) and `sig_vs_qrranker.md` (`BASE=...` to pair against another system). The two stages run
on their own from the cached replies:

```bash
python scripts/downstream/locomo_qa.py report [--llm gpt-4o-mini,gpt-5-mini,gpt-6-luna]
python scripts/downstream/locomo_qa.py sig --base qrranker
```
