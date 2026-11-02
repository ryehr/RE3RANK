"""The training mixture (paths.TRAIN_MIX): 38,000 items from five sources.

  MuSiQue-train 12,000 + NarrativeQA-train 12,000 + ConvoMem 7,000   (from work/mix/mix_convomem_squad.json, SQuAD items removed)
  QAConv 5,000 + FriendsQA 2,000                                     (from work/mix/mix_qaconv_friendsqa_squad.json)
concatenated in that order and shuffled with random.Random(0).

Inputs come from scripts/data/build_qaconv_friendsqa_squad.py (mix) and scripts/data/build_convomem.py
(assemble). CPU only:
  python scripts/data/build_train_mix.py
"""
import argparse
import collections
import random

from re3rank import paths

from _data_common import load_json, save_json, work


def build(with_convomem, with_qaconv_friendsqa):
    keep = [x for x in with_convomem if not str(x["id"]).startswith("squad-")]
    dial = [x for x in with_qaconv_friendsqa if str(x["id"]).startswith(("qaconv-", "friends-"))]
    mix = keep + dial
    random.Random(0).shuffle(mix)
    assert len(mix) == 38000 and len(set(x["id"] for x in mix)) == 38000
    return mix


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convomem-mix", default=work("mix", "mix_convomem_squad.json"))
    ap.add_argument("--qaconv-friendsqa-mix", default=work("mix", "mix_qaconv_friendsqa_squad.json"))
    ap.add_argument("--out", default=paths.data(paths.TRAIN_MIX))
    args = ap.parse_args()

    mix = build(load_json(args.convomem_mix), load_json(args.qaconv_friendsqa_mix))
    src = collections.Counter(
        next((p for p in ("nqa", "cvm", "qaconv", "friends") if str(x["id"]).startswith(p + "-")),
             "musique") for x in mix)
    print("composition:", dict(src))
    save_json(mix, args.out)
    print(f"{len(mix)} items -> {args.out}")


if __name__ == "__main__":
    main()
