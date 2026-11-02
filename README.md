# RE³Rank

Code for **RE³Rank**, a reranker that reads the attention of a few retrieval heads from an LLM's own
reasoning trace. <!-- TODO: paper title, authors, venue / arXiv link -->

> **Status.** This repository contains the code only. The processed datasets, reasoning traces,
> trained checkpoints and experimental results are not released here (results are reported in the
> paper); every dataset can be rebuilt from its public source with the scripts in `scripts/data/`
> (see [docs/DATA.md](docs/DATA.md)).

## Method

RE³Rank reranks the top-50 candidates of a first-stage retriever (Qwen3-Embedding-8B in all our
experiments).

1. **Generator.** A Qwen3 model in thinking mode reads the candidates and the question, with a plain
   system prompt ("You are a helpful assistant."): it is never told to rerank. It writes a reasoning
   trace, as it would to answer the question.
2. **Scorer.** A second copy of the backbone reads the same prompt followed by the trace. For a few
   selected attention heads, the score of a candidate is the attention mass that the trace tokens put
   on the candidate's tokens, averaged over the trace tokens and over the heads. The candidates are
   ranked by this score.
3. **Joint training.** The ranking loss on the scores (grouped contrastive loss + 2 × a Recall@k
   surrogate at k = 3, 10) trains the scorer directly and the generator through a straight-through
   estimator. The scorer's input embedding of every sampled trace token is
   `E[u_t] + p_t·E − sg(p_t·E)`, where `p_t` is the generator's softmax at that step: the forward
   value is the embedding of the sampled token, and the gradient flows into the generator's logits.

## Repository layout

```
re3rank/            shared code: prompts, the head scorer and ranking loss, metrics, paths, backbone adapter
scripts/data/       building the evaluation sets and the training mixture (docs/DATA.md)
scripts/heads/      scanning all attention heads of a backbone to select the scoring heads
scripts/train/      joint training of generator and scorer
scripts/eval/       reasoning-trace generation (vLLM), scoring, the nine-set evaluation
scripts/baselines/  external rerankers on the same candidates (docs/BASELINES.md)
scripts/downstream/ LoCoMo question answering from the reranked top-3
```

## Setup

```bash
conda create -n re3rank python=3.12 -y && conda activate re3rank
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
```

Paths are set by environment variables (defaults in brackets):

| variable | meaning |
|---|---|
| `RE3RANK_DATA` | datasets: raw downloads and the built candidate files (`./data`) |
| `RE3RANK_OUT` | checkpoints, traces, scores and reports (`./outputs`) |
| `HF_HOME` | Hugging Face cache for model weights and datasets |
| `OPENAI_API_KEY` | only for the NarrativeQA evidence labels (training and test) and the LoCoMo QA experiment |

The official QRRanker baseline needs an older `transformers`; see
[requirements-qrranker.txt](requirements-qrranker.txt).

All experiments ran on one node with 8 × NVIDIA H200 (141 GB).

## Data

Every item is a question with the 50 candidates of the first-stage retriever, in retrieval order:

```json
{"id": "...", "question": "...",
 "paragraphs": [{"title": "...", "paragraph_text": "...", "is_supporting": true}, "... 50 candidates"]}
```

**Evaluation (9 sets, 9,205 scored queries).**

| set | items | queries with a gold candidate in the top-50 |
|---|---|---|
| HotpotQA | 1,000 | 999 |
| 2WikiMultihopQA | 1,000 | 1,000 |
| MuSiQue | 1,000 | 1,000 |
| NQ | 1,000 | 998 |
| PopQA | 1,000 | 1,000 |
| NarrativeQA (HELMET) | 1,902 | 1,736 |
| DetectiveQA (English) | 600 | 504 |
| LoCoMo | 1,531 | 1,502 |
| LongMemEval | 466 | 466 |

**Training.** 38,000 items from NarrativeQA (12,000; silver evidence labels), MuSiQue (12,000),
ConvoMem (7,000), QAConv (5,000) and FriendsQA (2,000), with candidate pools retrieved the same way.
Training and evaluation do not overlap: MuSiQue shares no questions, the NarrativeQA training and
test books come from different splits, and LoCoMo and LongMemEval are never used for training.

[docs/DATA.md](docs/DATA.md) lists the sources and the build commands. Retrieval needs a GPU; the
NarrativeQA silver evidence labels (training and test) need the OpenAI API.

## Selecting the scoring heads

```bash
BASE=Qwen/Qwen3-4B bash scripts/heads/run_scan.sh
```

The base model writes traces for a seed sample (400 items each from MuSiQue and NarrativeQA), and one
forward pass per item measures, for every head, how much attention the trace puts on the gold
candidates. The report ranks heads by this gold attention mass, and `head_pairs.py` ranks single
heads and head pairs by the R@3 they reach without training. The heads used in the paper:

| backbone | heads (layer-head) | rule |
|---|---|---|
| Qwen3-0.6B | 20-4, 16-8 | best untrained R@3 pair |
| Qwen3-4B | 23-10, 21-8 | top-2 gold attention mass |
| Qwen3-8B | 15-11, 23-10 | best untrained R@3 pair |

## Training

```bash
SIZE=4b bash scripts/train/run_train.sh      # or 0.6b, 8b
```

| setting | value |
|---|---|
| initialisation | both models from the base Qwen3 checkpoint |
| data | one epoch of the training mixture: 37,759 items (prompts over 13k tokens skipped), 394 updates |
| batch | 96 items per update (8 ranks × 12; at 8B 4 ranks × 24 with the scorer on a second GPU) |
| trace sampling | temperature 1.0, at most 1,024 tokens, thinking block forced open |
| loss | CE + 2 × Recall@k surrogate (k = 3, 10; τ₁ = τ₂ = 1) on max-min normalised scores |
| optimiser | 8-bit paged AdamW, lr 1e-5 for both models, grad-norm clip 1.0 |

Checkpoints go to `$RE3RANK_OUT/ckpt/re3rank-<size>/upd{100,200,300}` and `final`. Each holds the
generator weights and `scorer.pt`, and a `SAVED` file marks it complete. The driver resumes from the
newest complete checkpoint.

## Evaluation

```bash
CKPT=outputs/ckpt/re3rank-4b/final BASE=Qwen/Qwen3-4B HEADS=23-10,21-8 bash scripts/eval/run_eval.sh
```

The script runs in two steps:
1. The trained generator writes one greedy trace per item (vLLM, at most 1,024 tokens).
2. The trained scorer ranks the candidates of each item.

Metrics are **conditional**: only queries with at least one gold candidate in the top-50 count, and
R@k divides by the number of gold candidates in the pool. NDCG@10 uses binary relevance. The summary
table gives each set and the macro average over the nine sets.

`scripts/eval/score.py` scores a single candidate file. With `--out` it also writes per-query scores;
these are the input to paired significance tests and to the downstream QA experiment.

## Baselines and downstream QA

[docs/BASELINES.md](docs/BASELINES.md) covers two groups of experiments:
- **Baselines.** Qwen3-Reranker, jina-reranker-v3.5, zerank-2, GroupRank, REARANK, ReasonRank and the
  official QRRanker, all on the same candidates.
- **LoCoMo QA.** Question answering on LoCoMo with GPT models reading the top-3 reranked chunks.

## Notes

- **Multi-GPU on cloud images.** Some cloud images preset NCCL network plugins that break
  single-node NCCL. If `torchrun` fails at the first collective, unset the `NCCL_*` variables, remove
  the plugin directory from `LD_LIBRARY_PATH`, and set `NCCL_NET_PLUGIN=none NCCL_TUNER_PLUGIN=none`.
- **libcuda.** If Triton cannot find `libcuda.so`, set `TRITON_LIBCUDA_PATH` to the driver's library
  directory.
- **Gemma-4.** `re3rank/backbone.py` also supports Gemma-4 backbones; the paper's results use Qwen3.

## Citation

<!-- TODO: BibTeX -->

## License

Apache-2.0 ([LICENSE](LICENSE)). `scripts/baselines/` contains functions copied from REARANK
(Apache-2.0) and ReasonRank (MIT); see [NOTICE](NOTICE).

## Acknowledgements

The scorer builds on query-focused retrieval heads (QRHead) and the QRRanker model. We use the Qwen3
models and the datasets listed in [docs/DATA.md](docs/DATA.md).
