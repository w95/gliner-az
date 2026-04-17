#!/usr/bin/env python3
"""Generate natural-prose Azerbaijani sentences with PII woven in.

The LLM (MiniMax M2.7) writes the prose; entity *values* are deterministic
(same inline generators as the synthetic script). After generation, a
deterministic span-finder extracts token positions. Samples where MiniMax
altered the entity value are discarded — we keep only verbatim matches.

Output: data/narrative_pii.json + push to {HF_USER}/azerbaijani-ner-narrative
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from hf_utils import push_gliner_dataset
from llm_client import call_claude

# Reuse the inline entity generators from the synthetic script — they're the
# source of truth for Azerbaijan-specific formats. Import them directly.
from generate_synthetic_az_ner_pattern_exhaustive import (
    gen_fin, gen_tin, gen_phone, gen_iban, gen_passport, gen_plate,
    gen_card, gen_email, gen_name, gen_city,
    tokenize, find_span,
)

random.seed(42)

SCENARIO_TYPES = [
    "news article about a business dispute",
    "customer service email",
    "HR onboarding document",
    "legal correspondence",
    "bank compliance report",
    "police incident report",
    "government service application confirmation",
    "insurance claim notification",
    "medical appointment reminder",
    "notary public statement",
    "rental agreement clause",
    "tax audit notice",
    "employment contract excerpt",
    "customer complaint letter",
    "tender application response",
    "internal memo from a bank branch",
    "court hearing notice",
    "utility bill reminder",
    "university enrolment confirmation",
    "shipping and customs declaration",
]

ENTITY_POOL: list[tuple[str, callable]] = [
    ("person",              gen_name),
    ("fin code",            gen_fin),
    ("tin",                 gen_tin),
    ("phone number",        gen_phone),
    ("iban",                gen_iban),
    ("passport number",     gen_passport),
    ("vehicle plate",       gen_plate),
    ("credit card number",  gen_card),
    ("email",               gen_email),
    ("location",            gen_city),
]

SYSTEM_PROMPT = (
    "You are an expert Azerbaijani writer producing realistic business and "
    "government documents. Your output language is Azerbaijani only.\n\n"
    "RULES:\n"
    "1. Write natural, native-quality Azerbaijani prose — use proper grammar, "
    "suffixes (-dir, -nin, -dən, -yə), and idiomatic phrasing.\n"
    "2. You MUST include every entity provided in the input EXACTLY AS WRITTEN — "
    "do not modify, translate, or reformat entity values.\n"
    "3. Weave entities naturally into the narrative — do not produce a bulleted list.\n"
    "4. Length: 2-4 sentences. Keep it concise but natural.\n"
    "5. Output ONLY the Azerbaijani text. No explanations, no JSON, no English, "
    "no preamble."
)


def build_user_prompt(scenario: str, entities: dict[str, str]) -> str:
    lines = "\n".join(f"- {label}: {value}" for label, value in entities.items())
    return (
        f"Write a short Azerbaijani paragraph (2-4 sentences) in the style of a {scenario}.\n\n"
        f"The paragraph MUST include these exact entities, unchanged:\n"
        f"{lines}\n\n"
        f"Remember: Write ONLY the Azerbaijani text. Include every entity exactly as given."
    )


def pick_entities() -> dict[str, str]:
    """Pick 2-4 distinct entity types and generate canonical values for each."""
    k = random.randint(2, 4)
    chosen = random.sample(ENTITY_POOL, k)
    return {label: gen() for label, gen in chosen}


def build_sample(narrative: str, entities: dict[str, str]) -> dict | None:
    """Tokenize narrative, verify every entity is present verbatim, return GLiNER record."""
    tokens = tokenize(narrative)
    spans: list[list] = []
    used: set[tuple[int, int]] = set()
    for label, value in entities.items():
        start, end = find_span(tokens, value)
        if start is None:
            return None  # entity was rewritten by the LLM — discard
        if (start, end) in used:
            return None
        used.add((start, end))
        spans.append([start, end, label])
    return {"tokenized_text": tokens, "ner": spans}


def generate_one() -> dict | None:
    scenario = random.choice(SCENARIO_TYPES)
    entities = pick_entities()
    narrative = call_claude(SYSTEM_PROMPT, build_user_prompt(scenario, entities))
    if not narrative:
        return None
    return build_sample(narrative.strip(), entities)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--repo-name", default="azerbaijani-ner-narrative")
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    samples: list[dict] = []
    max_attempts = args.num_samples * 3  # expect ~50% success rate
    submitted = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        in_flight = {pool.submit(generate_one) for _ in range(min(args.concurrency * 2, max_attempts))}
        submitted = len(in_flight)

        while in_flight and len(samples) < args.num_samples:
            done_iter = as_completed(in_flight)
            fut = next(done_iter)
            in_flight.discard(fut)
            try:
                s = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[narrative] worker error: {e}")
                s = None
            if s:
                samples.append(s)
                if len(samples) % 100 == 0:
                    print(f"  {len(samples)}/{args.num_samples} "
                          f"(success rate: {len(samples)/submitted:.1%})")
            # Refill
            if submitted < max_attempts and len(samples) < args.num_samples:
                in_flight.add(pool.submit(generate_one))
                submitted += 1

    print(f"\nDone. Kept {len(samples)} / {submitted} attempts "
          f"({len(samples)/max(submitted,1):.1%} success).")

    path = os.path.join(args.output_dir, "narrative_pii.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"Saved → {path}")

    if not args.no_push:
        push_gliner_dataset(samples, args.repo_name, private=True)


if __name__ == "__main__":
    main()
