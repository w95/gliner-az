#!/usr/bin/env python3
"""Merge all data sources into the final GLiNER training set, validate, push.

Sources (any missing ones are skipped with a warning):
- LocalDoc cleaned (or fallback to uncleaned)
- WikiANN
- Synthetic pattern-exhaustive
- Narrative PII (MiniMax-generated)

Output: data/{train,validation,test}_final.json (85/10/5)
        + push to {HF_USER}/azerbaijani-ner-final with 3 splits.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from hf_utils import push_gliner_dataset

random.seed(42)

SOURCES = [
    ("localdoc",   ["data/localdoc_train_cleaned.json", "data/localdoc_train.json"]),
    ("wikiann",    ["data/wikiann_train.json"]),
    ("synthetic",  ["data/synthetic_az_pattern_exhaustive.json"]),
    ("narrative",  ["data/narrative_pii.json"]),
]


def load_first_existing(paths: list[str]) -> list[dict] | None:
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def audit_counts(samples: list[dict], *, threshold: int = 2500,
                 required_labels: tuple[str, ...] = ("fin code", "tin", "phone number", "iban")) -> list[str]:
    counts: Counter = Counter()
    for s in samples:
        for _, _, label in s["ner"]:
            counts[label] += 1

    print(f"\n{'Entity':25s} {'Count':>8s}")
    print("-" * 35)
    for label in sorted(counts):
        print(f"{label:25s} {counts[label]:>8d}")

    issues = []
    for label in required_labels:
        if counts.get(label, 0) < threshold:
            issues.append(f"{label}: {counts.get(label, 0)} (<{threshold})")
    return issues


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data")
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--repo-name", default="azerbaijani-ner-final")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    args = p.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    all_samples: list[dict] = []
    for name, paths in SOURCES:
        data = load_first_existing(paths)
        if data is None:
            print(f"[merge] skipping '{name}' — none of {paths} exist")
            continue
        print(f"[merge] loaded '{name}': {len(data)} samples")
        all_samples.extend(data)

    if not all_samples:
        raise SystemExit("No data loaded. Run the data-prep scripts first.")

    if not args.skip_audit:
        issues = audit_counts(all_samples)
        if issues:
            print("\nWARN — patterned entities below threshold:")
            for i in issues:
                print(" -", i)
            print("Continue anyway (-y) or regenerate synthetic data.")

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": all_samples[:n_train],
        "validation": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    for name, data in splits.items():
        path = os.path.join(args.output_dir, f"{name}_final.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Wrote {name}: {len(data)} → {path}")

    if not args.no_push:
        push_gliner_dataset(splits, args.repo_name, private=True)


if __name__ == "__main__":
    main()
