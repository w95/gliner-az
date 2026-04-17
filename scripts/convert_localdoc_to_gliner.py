#!/usr/bin/env python3
"""Convert LocalDoc/azerbaijani-ner-dataset → GLiNER span JSON, push to HF Hub.

Source: https://huggingface.co/datasets/LocalDoc/azerbaijani-ner-dataset
Output: {HF_USER}/azerbaijani-ner-localdoc-gliner (and local data/localdoc_{split}.json)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
from collections import Counter

from datasets import load_dataset

from hf_utils import push_gliner_dataset

LABEL_MAP = {
    0: None, 1: "person", 2: "location", 3: "organisation", 4: "date",
    5: "time", 6: "money", 7: "percentage", 8: "facility", 9: "product",
    10: "event", 11: "art", 12: "law", 13: "language", 14: "gpe",
    15: "norp", 16: "ordinal", 17: "cardinal", 18: "disease", 19: "contact",
    20: "adage", 21: "quantity", 22: "miscellaneous", 23: "position", 24: "project",
}


def convert_sample(tokens: list[str], tags: list[int]) -> dict | None:
    """Flat integer tags → GLiNER span format. Consecutive same-tag tokens merge
    (LocalDoc has no BIO prefix, so adjacent same-type entities collapse)."""
    if len(tokens) != len(tags):
        return None

    entities: list[list] = []
    start_idx = None
    current_label = None

    for i, tag in enumerate(tags):
        label = LABEL_MAP.get(tag)
        if label is not None:
            if label == current_label:
                continue
            if current_label is not None:
                entities.append([start_idx, i - 1, current_label])
            start_idx = i
            current_label = label
        else:
            if current_label is not None:
                entities.append([start_idx, i - 1, current_label])
            current_label = None
            start_idx = None

    if current_label is not None:
        entities.append([start_idx, len(tags) - 1, current_label])

    if not entities:
        return None
    return {"tokenized_text": tokens, "ner": entities}


def valid(sample: dict) -> bool:
    n = len(sample["tokenized_text"])
    for start, end, label in sample["ner"]:
        if start < 0 or end >= n or start > end or not isinstance(label, str) or not label:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data")
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-miscellaneous", action="store_true",
                   help="Drop samples whose only entity is MISCELLANEOUS")
    p.add_argument("--repo-name", default="azerbaijani-ner-localdoc-gliner")
    p.add_argument("--no-push", action="store_true", help="Skip Hub push (local only)")
    args = p.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    print("Loading LocalDoc/azerbaijani-ner-dataset...")
    ds = load_dataset("LocalDoc/azerbaijani-ner-dataset")

    print("Converting to GLiNER format...")
    out: list[dict] = []
    skipped = 0
    counts: Counter = Counter()

    for sample in ds["train"]:
        raw_tokens, raw_tags = sample["tokens"], sample["ner_tags"]
        if raw_tokens is None or raw_tags is None:
            skipped += 1
            continue
        try:
            tokens = ast.literal_eval(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
            tags = ast.literal_eval(raw_tags) if isinstance(raw_tags, str) else raw_tags
        except (ValueError, SyntaxError):
            skipped += 1
            continue
        converted = convert_sample(tokens, tags)
        if converted is None:
            skipped += 1
            continue
        if args.skip_miscellaneous and not [e for e in converted["ner"] if e[2] != "miscellaneous"]:
            skipped += 1
            continue
        if not valid(converted):
            skipped += 1
            continue
        out.append(converted)
        for _, _, label in converted["ner"]:
            counts[label] += 1

    print(f"\nConverted: {len(out)}  |  Skipped: {skipped}  |  Input: {len(ds['train'])}")
    print("Entity distribution:")
    for label, c in counts.most_common():
        print(f"  {label:20s} {c:>8d}")

    random.Random(args.seed).shuffle(out)
    n = len(out)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": out[:n_train],
        "validation": out[n_train:n_train + n_val],
        "test": out[n_train + n_val:],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    for name, data in splits.items():
        path = os.path.join(args.output_dir, f"localdoc_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Wrote {name}: {len(data)} → {path}")

    if not args.no_push:
        push_gliner_dataset(splits, args.repo_name, private=True)


if __name__ == "__main__":
    main()
