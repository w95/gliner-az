#!/usr/bin/env python3
"""Drop LocalDoc samples flagged INCORRECT by the audit and push a cleaned split.

Input:  data/localdoc_train.json (or HF fallback) + data/audit_report.json
Output: data/localdoc_train_cleaned.json + push to {HF_USER}/azerbaijani-ner-localdoc-cleaned
"""
from __future__ import annotations

import argparse
import json
import os

from hf_utils import push_gliner_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", default="data/localdoc_train.json")
    p.add_argument("--audit-path", default="data/audit_report.json")
    p.add_argument("--output-path", default="data/localdoc_train_cleaned.json")
    p.add_argument("--repo-name", default="azerbaijani-ner-localdoc-cleaned")
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()

    with open(args.input_path) as f:
        train = json.load(f)
    with open(args.audit_path) as f:
        audit = json.load(f)

    exclude = set(audit.get("exclude_indices", []))
    cleaned = [s for i, s in enumerate(train) if i not in exclude]

    print(f"Original: {len(train)} | Cleaned: {len(cleaned)} | Removed: {len(train) - len(cleaned)}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False)
    print(f"Wrote {args.output_path}")

    if not args.no_push:
        push_gliner_dataset(cleaned, args.repo_name, private=True)


if __name__ == "__main__":
    main()
