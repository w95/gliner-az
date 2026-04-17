#!/usr/bin/env python3
"""Convert WikiANN Azerbaijani (IOB2) → GLiNER span JSON, push to HF Hub.

Source: unimelb-nlp/wikiann, config 'az'.
Output: {HF_USER}/azerbaijani-ner-wikiann-gliner (and local data/wikiann_{split}.json)
"""
from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset

from hf_utils import push_gliner_dataset

# WikiANN IOB2 indices → coarse label. 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC.
IOB_LABEL = {0: None, 1: "person", 2: "person", 3: "organisation",
             4: "organisation", 5: "location", 6: "location"}
B_TAGS = {1, 3, 5}
I_TAGS = {2, 4, 6}


def convert(tokens: list[str], tags: list[int]) -> dict | None:
    entities: list[list] = []
    start_idx: int | None = None
    current_label: str | None = None

    for i, tag in enumerate(tags):
        label = IOB_LABEL.get(tag)
        if tag in B_TAGS:
            if current_label is not None:
                entities.append([start_idx, i - 1, current_label])
            start_idx = i
            current_label = label
        elif tag in I_TAGS and label == current_label:
            continue
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data")
    p.add_argument("--repo-name", default="azerbaijani-ner-wikiann-gliner")
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()

    print("Loading unimelb-nlp/wikiann (az)...")
    ds = load_dataset("unimelb-nlp/wikiann", "az")

    os.makedirs(args.output_dir, exist_ok=True)
    splits: dict[str, list[dict]] = {}
    for split in ["train", "validation", "test"]:
        converted = [c for s in ds[split] if (c := convert(s["tokens"], s["ner_tags"]))]
        splits[split] = converted
        path = os.path.join(args.output_dir, f"wikiann_{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False)
        print(f"WikiANN {split}: {len(converted)} → {path}")

    if not args.no_push:
        push_gliner_dataset(splits, args.repo_name, private=True)


if __name__ == "__main__":
    main()
