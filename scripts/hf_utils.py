"""Tiny HF Hub helpers shared by data-prep scripts."""
from __future__ import annotations

import os
from typing import Optional

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def resolve_namespace() -> str:
    """Return the HF namespace to publish under.

    Priority: HF_USERNAME env var → `whoami()` lookup. Errors if neither works.
    """
    override = os.environ.get("HF_USERNAME")
    if override:
        return override
    token = os.environ.get("HF_TOKEN")
    try:
        info = HfApi().whoami(token=token) if token else HfApi().whoami()
    except Exception as e:
        raise RuntimeError(
            "Could not resolve HF namespace. Run `huggingface-cli login` "
            "or set HF_USERNAME / HF_TOKEN in .env."
        ) from e
    name = info.get("name")
    if not name:
        raise RuntimeError(f"whoami() returned no 'name' field: {info}")
    return name


def _ner_triples_to_dicts(samples: list[dict]) -> list[dict]:
    """Convert `ner: [[start, end, label], ...]` triples → list of dicts.

    Needed because pyarrow can't infer a schema for mixed-type lists.
    """
    out = []
    for s in samples:
        ner_dicts = [
            {"start": int(t[0]), "end": int(t[1]), "label": str(t[2])}
            for t in s["ner"]
        ]
        out.append({"tokenized_text": list(s["tokenized_text"]), "ner": ner_dicts})
    return out


def ner_dicts_to_triples(row: dict) -> dict:
    """Reverse of `_ner_triples_to_dicts`. Use when loading from Hub."""
    return {
        "tokenized_text": row["tokenized_text"],
        "ner": [[e["start"], e["end"], e["label"]] for e in row["ner"]],
    }


def push_gliner_dataset(
    samples: list[dict] | dict[str, list[dict]],
    repo_name: str,
    private: bool = True,
    namespace: Optional[str] = None,
) -> str:
    """Push GLiNER-format samples to HF Hub.

    - `samples` is either a flat list (→ single `train` split) or a dict of split-name → list.
    - `repo_name` is the repo slug without namespace (e.g. `azerbaijani-ner-synthetic`).
    - `ner` is serialised as a list of `{start, end, label}` dicts on the Hub.
      Use `ner_dicts_to_triples` when loading back.
    - Returns the full `namespace/repo_name` path.
    """
    ns = namespace or resolve_namespace()
    repo_id = f"{ns}/{repo_name}"

    if isinstance(samples, list):
        ds = DatasetDict({"train": Dataset.from_list(_ner_triples_to_dicts(samples))})
    else:
        ds = DatasetDict({
            k: Dataset.from_list(_ner_triples_to_dicts(v)) for k, v in samples.items()
        })

    ds.push_to_hub(repo_id, private=private)
    print(f"Pushed {sum(len(d) for d in ds.values())} samples → https://huggingface.co/datasets/{repo_id}")
    return repo_id
