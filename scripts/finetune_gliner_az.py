#!/usr/bin/env python3
"""Fine-tune GLiNER multi-v2.1 on Azerbaijani NER (Option A hyperparameters).

Pulls data from {HF_USER}/azerbaijani-ner-final on the Hub (or local JSON fallback),
fine-tunes with focal_loss_gamma=2.5 + fp16 on T4/4090, and pushes a checkpoint
to {HF_USER}/gliner-azerbaijani-ner-v1 after every epoch so a Colab disconnect
costs at most one epoch of work.

Usage (local / RunPod):
    python scripts/finetune_gliner_az.py --epochs 10

Colab:
    See notebooks/finetune_colab.ipynb — it wraps this script.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from gliner import GLiNER  # noqa: E402
from gliner.data_processing.collator import SpanDataCollator  # noqa: E402
from gliner.training import Trainer, TrainingArguments  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.utils import RepositoryNotFoundError  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from hf_utils import ner_dicts_to_triples, resolve_namespace  # noqa: E402

load_dotenv()

# -------- default labels (Option A entity catalog) ---------
DEFAULT_LABELS = [
    # Patterned PII
    "fin code", "tin", "phone number", "iban", "passport number",
    "vehicle plate", "credit card number", "email", "postal code",
    # Semantic
    "person", "organisation", "location", "gpe", "date", "time", "money",
    "percentage", "facility", "product", "event", "position", "law",
    "language", "norp", "disease", "project",
]
PATTERNED = {"fin code", "tin", "phone number", "iban", "passport number",
             "vehicle plate", "credit card number", "email", "postal code"}


# ============================================================
# Data loading — Hub primary, local fallback
# ============================================================

def load_splits(namespace: str, repo_name: str, local_dir: str) -> dict[str, list[dict]]:
    repo_id = f"{namespace}/{repo_name}"
    try:
        print(f"[data] loading {repo_id} from HF Hub...")
        ds = load_dataset(repo_id)
        return {k: [ner_dicts_to_triples(r) for r in ds[k]] for k in ds}
    except Exception as e:  # noqa: BLE001
        print(f"[data] hub load failed ({e}); falling back to {local_dir}/*.json")
        out = {}
        for split, fname in [("train", "train_final.json"),
                              ("validation", "validation_final.json"),
                              ("test", "test_final.json")]:
            path = os.path.join(local_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    out[split] = json.load(f)
        if "train" not in out or "validation" not in out:
            raise SystemExit("No training data found locally or on Hub.")
        return out


# ============================================================
# Resume — check Hub for existing checkpoint + epoch count
# ============================================================

def resolve_resume(repo_id: str) -> tuple[str, int]:
    """Return (base-model-id-to-load-from, epochs-already-done).

    If `repo_id` exists on the Hub, load from there and read training_state.json
    for the completed-epoch count. Otherwise start from the base multilingual model.
    """
    base = "urchade/gliner_multi-v2.1"
    api = HfApi()
    try:
        api.repo_info(repo_id, repo_type="model")
    except RepositoryNotFoundError:
        print(f"[resume] no prior run at {repo_id}; starting from {base}")
        return base, 0
    except Exception as e:  # noqa: BLE001
        print(f"[resume] hub check failed ({e}); starting from {base}")
        return base, 0

    try:
        state_file = hf_hub_download(repo_id, "training_state.json", repo_type="model")
        with open(state_file) as f:
            state = json.load(f)
        epochs_done = int(state.get("epochs_completed", 0))
    except Exception:
        epochs_done = 0

    print(f"[resume] resuming from {repo_id} (epochs_done={epochs_done})")
    return repo_id, epochs_done


# ============================================================
# Per-epoch Hub checkpoint callback
# ============================================================

class HubEpochCheckpoint(TrainerCallback):
    def __init__(self, model: GLiNER, repo_id: str, start_epoch: int, private: bool = True):
        self.model = model
        self.repo_id = repo_id
        self.start_epoch = start_epoch
        self.private = private

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_done = self.start_epoch + int(state.epoch or 0)
        print(f"[checkpoint] pushing epoch {epoch_done} to {self.repo_id}...")
        try:
            self.model.push_to_hub(self.repo_id, private=self.private)
        except Exception as e:  # noqa: BLE001
            print(f"[checkpoint] push failed: {e}")
            return

        # Push sidecar training_state.json via HfApi
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                json.dump({"epochs_completed": epoch_done}, tf)
                state_path = tf.name
            HfApi().upload_file(
                path_or_fileobj=state_path,
                path_in_repo="training_state.json",
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"training_state after epoch {epoch_done}",
            )
            os.unlink(state_path)
        except Exception as e:  # noqa: BLE001
            print(f"[checkpoint] state upload failed: {e}")


# ============================================================
# Per-entity F1 evaluation (sampled to stay fast)
# ============================================================

def evaluate_per_entity(model: GLiNER, samples: list[dict], labels: list[str],
                        threshold: float = 0.4) -> dict:
    model.eval()
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for s in samples:
        tokens = s["tokenized_text"]
        text = " ".join(tokens)
        gold = {(" ".join(tokens[a:b + 1]).lower(), lbl) for a, b, lbl in s["ner"]}
        try:
            preds = model.predict_entities(text, labels, threshold=threshold)
        except Exception:
            continue
        pred = {(p["text"].lower(), p["label"]) for p in preds}
        for label in labels:
            g = {e for e in gold if e[1] == label}
            p = {e for e in pred if e[1] == label}
            per_type[label]["tp"] += len(p & g)
            per_type[label]["fp"] += len(p - g)
            per_type[label]["fn"] += len(g - p)

    results: dict[str, dict] = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for label, c in per_type.items():
        total["tp"] += c["tp"]; total["fp"] += c["fp"]; total["fn"] += c["fn"]
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        results[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    mp = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    mr = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
    results["_micro"] = {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(mf, 4)}
    return results


class PerEntityF1Callback(TrainerCallback):
    def __init__(self, model: GLiNER, val_samples: list[dict], labels: list[str],
                 max_samples: int = 500, patterned_labels: set[str] = PATTERNED):
        self.model = model
        # Subsample val to keep eval cheap on T4
        rng = random.Random(0)
        self.samples = rng.sample(val_samples, min(max_samples, len(val_samples)))
        self.labels = labels
        self.patterned_labels = patterned_labels
        self.best_patterned_f1 = 0.0
        self.epochs_without_improvement = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        res = evaluate_per_entity(self.model, self.samples, self.labels)
        pat_f1s = [res[lbl]["f1"] for lbl in self.patterned_labels if lbl in res]
        pat_f1 = sum(pat_f1s) / len(pat_f1s) if pat_f1s else 0.0
        micro = res["_micro"]["f1"]
        print(f"\n[eval @ epoch {state.epoch:.0f}] micro-F1={micro:.4f}  patterned-F1={pat_f1:.4f}")
        for lbl in sorted(res):
            if lbl == "_micro":
                continue
            r = res[lbl]
            print(f"  {lbl:25s} p={r['precision']:.3f} r={r['recall']:.3f} f1={r['f1']:.3f}")

        if pat_f1 > self.best_patterned_f1 + 1e-4:
            self.best_patterned_f1 = pat_f1
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1


# ============================================================
# Main
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-repo", default="azerbaijani-ner-final",
                   help="HF dataset repo name under the user's namespace.")
    p.add_argument("--model-repo", default="gliner-azerbaijani-ner-v1",
                   help="HF model repo name to publish checkpoints into.")
    p.add_argument("--epochs", type=int, default=10,
                   help="Total target epochs (Option A: 7-10).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-6, help="Encoder LR")
    p.add_argument("--others-lr", type=float, default=2e-5, help="Task head LR")
    p.add_argument("--focal-gamma", type=float, default=2.5)
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--output-dir", default="./checkpoints/azerbaijani-gliner")
    p.add_argument("--local-data-dir", default="./data")
    p.add_argument("--max-val-eval", type=int, default=500,
                   help="Sub-sample val set during per-epoch eval.")
    p.add_argument("--no-push", action="store_true",
                   help="Skip Hub checkpointing (for dry runs).")
    p.add_argument("--public", action="store_true",
                   help="Publish checkpoint repo public (default: private).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] using {device}")

    namespace = resolve_namespace()
    model_repo_id = f"{namespace}/{args.model_repo}"

    # 1. Resolve resume
    base_model_id, epochs_done = resolve_resume(model_repo_id)
    remaining_epochs = max(1, args.epochs - epochs_done)
    if epochs_done >= args.epochs:
        print(f"[resume] already trained {epochs_done}/{args.epochs} — nothing to do.")
        return

    # 2. Load model
    print(f"[model] loading {base_model_id}...")
    model = GLiNER.from_pretrained(base_model_id)
    model.to(device)

    # 3. Load data
    splits = load_splits(namespace, args.data_repo, args.local_data_dir)
    train_data = splits["train"]
    eval_data = splits.get("validation") or splits.get("test") or []
    print(f"[data] train={len(train_data)}  val={len(eval_data)}")

    # 4. Collator — GLiNER ≥0.2.27 renamed `DataCollator` to `SpanDataCollator`
    # (span-type matching is what the multi-v2.1 backbone uses).
    data_collator = SpanDataCollator(
        model.config, data_processor=model.data_processor, prepare_labels=True,
    )

    # 5. Training args
    use_fp16 = device == "cuda"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        others_lr=args.others_lr,
        weight_decay=args.weight_decay,
        others_weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        focal_loss_alpha=args.focal_alpha,
        focal_loss_gamma=args.focal_gamma,
        num_train_epochs=remaining_epochs,
        save_strategy="no",         # we push per-epoch via callback instead
        logging_steps=50,
        dataloader_num_workers=2,
        fp16=use_fp16,
        report_to="none",
    )

    # 6. Callbacks
    callbacks: list[TrainerCallback] = []
    callbacks.append(PerEntityF1Callback(model, eval_data, DEFAULT_LABELS,
                                         max_samples=args.max_val_eval))
    if not args.no_push:
        callbacks.append(HubEpochCheckpoint(model, model_repo_id,
                                            start_epoch=epochs_done,
                                            private=not args.public))

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # 8. Train
    print(f"[train] starting ({remaining_epochs} epochs remaining of {args.epochs})")
    trainer.train()

    # 9. Final save (local + hub if enabled)
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    print(f"[save] wrote {final_dir}")
    if not args.no_push:
        try:
            model.push_to_hub(model_repo_id, private=not args.public)
            print(f"[save] pushed final → https://huggingface.co/{model_repo_id}")
        except Exception as e:  # noqa: BLE001
            print(f"[save] final push failed: {e}")


if __name__ == "__main__":
    main()
