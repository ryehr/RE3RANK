# Data

The scripts in `scripts/data/` rebuild, from public sources, the nine evaluation candidate files and
the 38,000-item training mixture. Every item is a question with the top-50 candidates of the first-stage
retriever, in retrieval order, each with a binary label:

```json
{"id": "...", "question": "...", "n_gold": 2, "gold_titles": [],
 "paragraphs": [{"idx": 17, "title": "...", "paragraph_text": "...", "is_supporting": true}, "..."]}
```

`n_gold` (all gold units of the question, also those outside the top-50) is present in every set
except HotpotQA and the NarrativeQA training items; a few sets carry extra fields (`answer`, `category`,
`qtype`, `retriever_score`). A candidate is positive if it contains any gold unit (passage, paragraph,
turn, message or answer string, depending on the source).

**First stage, everywhere.** Qwen3-Embedding-8B (bf16, left padding, normalised embeddings, cosine
similarity), the question encoded with the model's `query` prompt, the candidate text without a prompt
(`title\ntext` for the HippoRAG 2 and MuSiQue-train passages, the chunk text elsewhere). All token
counts, chunk budgets and message caps use the Qwen3 tokenizer (`Qwen/Qwen3-4B-Instruct-2507`).

## Layout

Everything lives under the data root `$RE3RANK_DATA` (default `<repo>/data`, `re3rank/paths.py`):

```
raw/locomo/locomo10.json              download_raw.sh (not on the HF Hub)
raw/qaconv/QAConv-V1.1/               download_raw.sh
raw/friendsqa/friendsqa_{trn,dev,tst}.json
work/<source>/                        intermediate files of the multi-stage builders
work/mix/                             the two intermediate training mixtures
candidates/<set>.json                 the nine evaluation sets (paths.EVAL_SETS)
candidates/train_mix.json             the training mixture (paths.TRAIN_MIX)
```

Hugging Face datasets are read by hub id through the HF cache (`$HF_HOME`), pinned to revisions
checked to reproduce the released files (the `HF_REVISION` constants in the scripts).

## Requirements

- The main environment of `requirements.txt` (`pip install -e .`), plus `unzip` and `curl` for
  `download_raw.sh`.
- **GPU** for every `embed` stage (Qwen3-Embedding-8B in bf16, about 16 GB) and for the top-k of
  `build_musique_train.py assemble`. All other stages run on CPU.
- **OpenAI API** for the NarrativeQA silver labels only (`build_narrativeqa.py silver`, default model
  `gpt-5.4`, `--model` to change). The key is read from `OPENAI_API_KEY`.
- The HELMET selection of the NarrativeQA test set counts tokens with the Llama-2 tokenizer
  (`NousResearch/Llama-2-7b-hf`, the same files as the gated `meta-llama/Llama-2-7b-hf`).

## Sources

| set | public source | builder | GPU | API | items |
|---|---|---|---|---|---|
| HotpotQA | HF `osunlp/HippoRAG_2` (`hotpotqa*.json`) | `build_hipporag.py` | yes | | 1,000 |
| 2WikiMultihopQA | HF `osunlp/HippoRAG_2` (`2wikimultihopqa*.json`) | `build_hipporag.py` | yes | | 1,000 |
| MuSiQue | HF `osunlp/HippoRAG_2` (`musique*.json`) | `build_hipporag.py` | yes | | 1,000 |
| NQ | HF `osunlp/HippoRAG_2` (`nq_rear*.json`) | `build_hipporag.py` | yes | | 1,000 |
| PopQA | HF `osunlp/HippoRAG_2` (`popqa*.json`) | `build_hipporag.py` | yes | | 1,000 |
| NarrativeQA (HELMET) | HF `deepmind/narrativeqa`, test split | `build_narrativeqa.py --split test` | yes | yes | 1,902 |
| DetectiveQA (English) | HF `Phospheneser/DetectiveQA` (`novel_data_en`, `anno_data_en`) | `build_detectiveqa.py` | yes | | 600 |
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) `data/locomo10.json` | `build_locomo.py` | yes | | 1,531 |
| LongMemEval | HF `xiaowu0162/longmemeval` (`longmemeval_s`) | `build_longmemeval.py` | yes | | 466 |

| training source | public source | builder | GPU | API | in the mixture |
|---|---|---|---|---|---|
| NarrativeQA-train | HF `deepmind/narrativeqa`, train split | `build_narrativeqa.py --split train` | yes | yes | 12,000 |
| MuSiQue-train | HF `dgslibisey/MuSiQue` (`musique_ans_v1.0_train.jsonl`) | `build_musique_train.py` | yes | | 12,000 |
| ConvoMem | HF `Salesforce/ConvoMem` (`core_benchmark`) | `build_convomem.py` | yes | | 7,000 |
| QAConv | [salesforce/QAConv](https://github.com/salesforce/QAConv) `dataset/QAConv-V1.1.zip` | `build_qaconv_friendsqa_squad.py` | yes | | 5,000 |
| FriendsQA | [emorynlp/FriendsQA](https://github.com/emorynlp/FriendsQA) `dat/friendsqa_{trn,dev,tst}.json` | `build_qaconv_friendsqa_squad.py` | yes | | 2,000 |
| SQuAD v1.1 | HF `rajpurkar/squad` | `build_qaconv_friendsqa_squad.py` | yes | | 0 (see below) |

## Build

`bash scripts/data/build_all.sh` runs every step below in order (GPU stages one shard per visible GPU,
logs in `$RE3RANK_DATA/logs/`; `PART=eval` or `PART=train` for one half, `DRY=1` to print the
commands). Step by step, from the repository root:

```bash
bash scripts/data/download_raw.sh            # LoCoMo, QAConv, FriendsQA; checks sha256

# --- evaluation sets
python scripts/data/build_hipporag.py --dataset all                        # GPU
python scripts/data/build_detectiveqa.py prep
python scripts/data/build_detectiveqa.py chunk
python scripts/data/build_detectiveqa.py embed                             # GPU (--shard/--nshards to split)
python scripts/data/build_detectiveqa.py assemble
python scripts/data/build_locomo.py                                        # GPU
python scripts/data/build_longmemeval.py parse
for s in 0 1 2 3; do python scripts/data/build_longmemeval.py embed --shard $s --nshards 4; done   # GPU
python scripts/data/build_longmemeval.py assemble

# --- NarrativeQA, both splits (they share docs.json, chunks.json and the book embeddings)
python scripts/data/build_narrativeqa.py prepare --split test              # Llama-2 tokenizer
python scripts/data/build_narrativeqa.py prepare --split train
python scripts/data/build_narrativeqa.py chunk
python scripts/data/build_narrativeqa.py embed --split test                # GPU
python scripts/data/build_narrativeqa.py embed --split train               # GPU (book embeddings are reused)
python scripts/data/build_narrativeqa.py silver --split test               # OpenAI API
python scripts/data/build_narrativeqa.py silver --split train              # OpenAI API
python scripts/data/build_narrativeqa.py assemble --split test             # -> candidates/narrativeqa.json
python scripts/data/build_narrativeqa.py assemble --split train            # -> work/narrativeqa/train_candidates.json

# --- training sources and the mixture
python scripts/data/build_musique_train.py prep
for s in 0 1 2; do python scripts/data/build_musique_train.py embed --shard $s --nshards 3; done  # GPU
python scripts/data/build_musique_train.py assemble --nshards 3            # GPU
python scripts/data/build_qaconv_friendsqa_squad.py parse
for src in qaconv friends squad; do python scripts/data/build_qaconv_friendsqa_squad.py embed --source $src; done  # GPU
python scripts/data/build_qaconv_friendsqa_squad.py mix                    # -> work/mix/mix_qaconv_friendsqa_squad.json
python scripts/data/build_convomem.py parse
python scripts/data/build_convomem.py split --nshards 4                    # optional, less RAM per shard
for s in 0 1 2 3; do python scripts/data/build_convomem.py embed --shard $s --nshards 4; done      # GPU
python scripts/data/build_convomem.py assemble                             # -> work/mix/mix_convomem_squad.json
python scripts/data/build_train_mix.py                                     # -> candidates/train_mix.json
```

Shards of one stage can run in parallel, one GPU each (`CUDA_VISIBLE_DEVICES=i`). Keep the shard counts
shown for LongMemEval (4), ConvoMem (4) and MuSiQue-train (3): the first two concatenate their shards
in shard order (item order, and for ConvoMem the sampling), the third embeds the corpus round-robin
per shard, as the released data was built.

## Evaluation sets

| set | pool (first-stage scope) | candidate unit | gold |
|---|---|---|---|
| HotpotQA | the whole corpus, 9,811 passages | a released passage | passages whose title is a `supporting_facts` title |
| 2WikiMultihopQA | 6,119 passages (deduplicated by title+text) | a released passage | the context paragraphs of the `supporting_facts` titles, matched by exact title+text |
| MuSiQue | 11,656 passages (deduplicated) | a released passage | the paragraphs marked `is_supporting`, by title+text |
| NQ | 9,633 passages (deduplicated) | a released passage | the contexts marked `is_supporting`, by title+text |
| PopQA | 8,676 passages (deduplicated) | a released passage | the paragraphs marked `is_supporting` (question entity and answer entity pages), by title+text |
| NarrativeQA | the question's book | 200-token chunk | silver labels (below) |
| DetectiveQA | the question's novel (62) | whole paragraphs packed to ≤ 200 tokens | chunks holding a `clue_position` paragraph |
| LoCoMo | the question's conversation (10) | turns of one session packed to ≤ 300 tokens | chunks holding an `evidence` turn (`dia_id`) |
| LongMemEval | the question's haystack (~50 sessions) | 2-message window | windows of an evidence session holding a `has_answer` message |

- **NarrativeQA (HELMET).** HELMET's `load_narrativeqa` rule: test-split questions whose raw book is
  longer than 131,072 Llama-2 tokens, 1,981 questions over 66 books. A book is cropped to the story body
  with the dataset's start/end markers and cut into non-overlapping 200-token chunks. Silver labels
  follow MiA-RAG (Alg. 1): the question Q, Q + answer and the answer each retrieve their top-10 chunks of
  the book, the pooled chunks are ranked by votes (ties by best rank) and the first 10 kept, and the
  OpenAI model selects the relevant ones from these 10 with the MiA-RAG Figure 12 prompt (first reference
  answer). The 79 questions with no selected chunk are dropped (1,902 remain, over 65 books; 1,736 have a
  gold chunk in their top-50). The candidates are retrieved with the question alone.
- **DetectiveQA.** All 600 English questions (154 `human_anno`, 446 `AIsup_anno`), question stem only.
  Paragraphs are never split, so each clue paragraph lies in exactly one chunk.
- **LoCoMo.** Every category except 5 (adversarial); 9 questions whose evidence turns cannot be found
  are dropped. Each chunk starts with `Session date: <date>`; a turn is `speaker: text`, plus
  ` [shares a photo: <caption>]` for a shared image.
- **LongMemEval.** LongMemEval-S without the 30 abstention questions (470); a session is cut into windows
  of 2 messages (`role: content`, content capped at 200 tokens, a last single message merged into the
  previous window), each prefixed with `[<session date>] `. An evidence session without any `has_answer`
  flag counts all its windows as gold (45 sessions). The 4 questions whose gold windows all miss the
  top-50 are dropped (466 remain). The first stage embeds the window text only; the candidate title is
  the window id `<session id>-w<k>`.

## Training mixture

| source | questions | usable (gold in the top-50) | in the mixture | pool |
|---|---|---|---|---|
| NarrativeQA-train | 20,000 drawn with `random.Random(42)`; 19,256 with silver labels | 18,287 | 12,000 | the question's book |
| MuSiQue-train | 19,938 (answerable train) | 19,800 | 12,000 | all train paragraphs, deduplicated (84,559) |
| ConvoMem | 9,100 drawn (3,900 / 2,600 / 1,950 / 650) | 3,777 / 1,622 / 1,789 / 637 | 3,378 / 1,622 / 1,500 / 500 | the persona's conversations |
| QAConv | 13,643 human-written, answerable | 8,942 | 5,000 | 31,978 chunks of all 18,728 segments |
| FriendsQA | 9,874 (train) | 6,421 | 2,000 | 2,397 chunks of all scenes |
| SQuAD | 12,000 of the deduplicated questions | 11,802 | 0 | 18,891 paragraphs |

(ConvoMem categories: changing / implicit / user / assistant.) "Usable" means at least one gold
candidate in the top-50 and, for the sources drawn in step 1, at least one non-gold one. The mixture is
built in three steps:

1. `build_qaconv_friendsqa_squad.py mix` draws MuSiQue 12,000 + NarrativeQA 12,000 + QAConv 5,000 +
   FriendsQA 2,000 + SQuAD 7,000 from the usable items (one `random.Random(0)` stream: each source
   shuffled in that order, then the concatenation);
2. `build_convomem.py assemble` draws the 7,000 ConvoMem items and swaps them in for QAConv and
   FriendsQA (`random.Random(0)`);
3. `build_train_mix.py` keeps the MuSiQue, NarrativeQA and ConvoMem items of step 2, appends the QAConv
   and FriendsQA items of step 1 and shuffles with `random.Random(0)`.

SQuAD ends up with no item in the final mixture, but it has to be built: its item count and positions in
steps 1 and 2 fix the order of everything else.

Per-source details:
- **NarrativeQA-train.** Chunking and silver labels as for the test set; questions without a silver
  chunk or without a silver chunk in their top-50 are dropped. These items carry no `n_gold`.
- **MuSiQue-train.** A candidate is gold if its title and text equal one of the question's supporting
  paragraphs.
- **ConvoMem.** Items with at least 2 evidence messages, each in its own conversation (abstention and
  preference items are not used); `random.Random(3)`, round-robin over personas. The pool of a persona is
  its filler conversations plus the evidence conversations of all its selected items, cut into windows
  of 6 messages (`speaker: text`, 200-token cap, a tail of fewer than 3 messages merged back). An
  evidence message is located by exact text, then by its first 80 characters, then by normalised text;
  its window is the gold. The candidate title is the window id (`ev<n>c<m>-w<k>` for evidence
  conversations, `fill<j>-w<k>` for filler conversations).
- **QAConv.** Turns `speaker: text` packed in order into chunks of ≤ 300 tokens (no turn split; an
  overlong turn cut to 300 tokens), never across segments; all splits form the pool, questions come from
  the train split only (no `QG` flag, not unanswerable). Gold: the chunks of the question's own segment
  whose normalised text contains a reference answer.
- **FriendsQA.** Same packing per scene paragraph; all splits form the pool, train questions only. Gold:
  the chunks holding an answer utterance.

## Reproducibility

- The CPU stages (parsing, selection, chunking, labelling, item assembly, the three mixing steps) are
  deterministic. Run with the versions of `requirements.txt` on the pinned sources, they reproduce the
  released files byte for byte, given the same top-50 indices and silver labels.
- The top-50 lists depend on bf16 embeddings; another GPU or library version can swap near-ties and
  change a few candidate lists.
- The silver labels come from an API model and are not reproducible call for call. To reuse a given
  set of labels, place them as `work/narrativeqa/silver_test.json` / `silver_train.json`
  (`{question index: [chunk indices]}`) and skip the `silver` stage.
