#!/usr/bin/env python3
"""Use MiniMax M2.7 to spot-check LocalDoc NER annotations.

For a random 500-sample subset, ask the LLM whether each annotation is correct.
Writes data/audit_report.json with:
- summary counts
- exclude_indices: sample indices flagged INCORRECT (to drop at merge time)
- detailed_results for inspection
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

from llm_client import call_claude_json

random.seed(42)

AUDIT_SYSTEM_PROMPT = (
    "You are an expert Azerbaijani linguist auditing NER annotations.\n\n"
    "You will be shown an Azerbaijani sentence with entity annotations. Your job "
    "is to identify whether each annotation is correct.\n\n"
    "Entity type definitions:\n"
    "- person: Human names (first, last, or full)\n"
    "- location: Physical locations, cities, countries\n"
    "- organisation: Companies, government bodies, institutions\n"
    "- gpe: Geopolitical entities (states, provinces, regions)\n"
    "- date: Calendar dates, days, months, years\n"
    "- time: Clock times\n"
    "- money: Currency amounts\n"
    "- percentage: Percentage values\n"
    "- facility: Buildings, airports, hospitals, specific facilities\n"
    "- product: Commercial products or services\n"
    "- event: Named events (conferences, wars, festivals)\n"
    "- position: Job titles, ranks\n"
    "- miscellaneous: Only when nothing else fits\n\n"
    "OUTPUT FORMAT: Respond with a JSON object:\n"
    "{\n"
    "  \"verdict\": \"CORRECT\" | \"INCORRECT\" | \"PARTIAL\",\n"
    "  \"issues\": [\n"
    "    {\"entity\": \"extracted text\", \"current_label\": \"X\", \"suggested_label\": \"Y\" | \"REMOVE\"}\n"
    "  ]\n"
    "}\n\n"
    "Only flag clear errors. If uncertain, mark as CORRECT."
)


def build_prompt(sample: dict) -> str:
    tokens = sample["tokenized_text"]
    text = " ".join(tokens)
    lines = [
        f"  - '{' '.join(tokens[start:end + 1])}' -> {label}"
        for start, end, label in sample["ner"]
    ]
    return (
        f"Sentence: {text}\n\n"
        f"Annotations:\n" + "\n".join(lines) + "\n\n"
        f"Are these annotations correct? Respond with JSON only."
    )


def audit_sample(sample: dict, sample_idx: int) -> dict | None:
    parsed = call_claude_json(AUDIT_SYSTEM_PROMPT, build_prompt(sample), max_tokens=512)
    if not parsed:
        return None
    return {
        "sample_idx": sample_idx,
        "verdict": parsed.get("verdict", "UNKNOWN"),
        "issues": parsed.get("issues", []),
    }


def _load_samples(input_path: str | None) -> list[dict]:
    """Load GLiNER-formatted samples. Falls back to local file if provided,
    else pulls the HF-pushed LocalDoc dataset (if available)."""
    if input_path and os.path.exists(input_path):
        with open(input_path) as f:
            return json.load(f)

    # Fall back to the LocalDoc conversion output pushed to HF.
    from hf_utils import ner_dicts_to_triples, resolve_namespace
    ns = resolve_namespace()
    repo = f"{ns}/azerbaijani-ner-localdoc-gliner"
    print(f"Loading {repo} from HF Hub...")
    ds = load_dataset(repo, split="train")
    return [ner_dicts_to_triples(r) for r in ds]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", default="data/localdoc_train.json",
                   help="Local GLiNER JSON; falls back to HF Hub if absent.")
    p.add_argument("--output-path", default="data/audit_report.json")
    p.add_argument("--audit-size", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    samples = _load_samples(args.input_path)
    print(f"Loaded {len(samples)} samples.")
    sample_indices = random.sample(range(len(samples)), min(args.audit_size, len(samples)))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(audit_sample, samples[i], i): i for i in sample_indices}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(sample_indices)}")
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[audit] worker error: {e}")
                continue
            if r:
                results.append(r)

    verdicts = {"CORRECT": 0, "INCORRECT": 0, "PARTIAL": 0, "UNKNOWN": 0}
    for r in results:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

    print("\n=== Audit Summary ===")
    print(f"Audited: {len(results)}")
    for v, c in verdicts.items():
        pct = c / max(len(results), 1) * 100
        print(f"  {v}: {c} ({pct:.1f}%)")

    exclude = [r["sample_idx"] for r in results if r["verdict"] == "INCORRECT"]

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": verdicts,
            "audit_size": len(results),
            "exclude_indices": exclude,
            "detailed_results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output_path}")
    print(f"Exclusions: {len(exclude)} samples flagged INCORRECT")

    err_rate = verdicts["INCORRECT"] / max(len(results), 1)
    print(f"Estimated dataset-wide error rate: {err_rate:.1%}")
    print(f"Projected bad samples in full set: ~{int(err_rate * len(samples))} of {len(samples)}")


if __name__ == "__main__":
    main()
