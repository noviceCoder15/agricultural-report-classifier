"""Fine-tune GPT-2 to classify reports by predicting a special label token.

Needs PyTorch and transformers (and accelerate, which Trainer uses).

Steps: (1) add one special token per label to the tokenizer, (2) resize the
model's embeddings to match, (3) train with the Hugging Face Trainer so that,
after 'Report: ... Category:', the model predicts the right label token, and
(4) classify by reading the model's scores for just the label tokens.
"""
from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import torch

from . import core, storage

DEFAULT_MODEL = "gpt2"


def load_base(model_name: str = DEFAULT_MODEL):
    """Load a pre-trained causal language model and its tokenizer (a name or a local folder)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:                 # GPT-2 has no padding token
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


# --------------------------------------------------------------------------
# Milestone: extend the tokenizer and the model with the label tokens
# --------------------------------------------------------------------------

def add_label_tokens(tokenizer, model, labels: list[str]) -> list[int]:
    """Add '<label>' special tokens and make room for them in the model. Returns their ids."""
    tokens = [core.label_token(label) for label in labels]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    model.resize_token_embeddings(len(tokenizer))
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("the label tokens could not be added to the tokenizer")
    return ids


# --------------------------------------------------------------------------
# Milestone: train with the Hugging Face Trainer
# --------------------------------------------------------------------------

class ReportDataset(torch.utils.data.Dataset):
    """Tokenized examples: {'input_ids': [...], 'labels': [...]}."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


class PadCollator:
    """Pads a batch to its longest example and returns tensors (labels padded with -100)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        batch = core.pad_batch(features, self.pad_id)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def build_items(tokenizer, records: list[dict], label_ids: dict[str, int], max_length: int) -> list[dict]:
    return [core.encode_example(tokenizer, r["text"], label_ids[r["label"]], max_length) for r in records]


def train_classifier(model, items: list[dict], pad_id: int, epochs: int = 4, batch_size: int = 8,
                     lr: float = 1e-4, seed: int = 1337, log_every: int = 10) -> dict:
    """Fine-tune the whole model with Trainer on tokenized examples."""
    from transformers import Trainer, TrainingArguments, set_seed

    if not items:
        raise ValueError("no training examples")
    set_seed(seed)
    total_steps = math.ceil(len(items) / batch_size) * epochs
    with tempfile.TemporaryDirectory() as work_dir:
        arguments = TrainingArguments(
            output_dir=work_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=max(1, total_steps // 10),
            logging_steps=log_every,
            save_strategy="no",
            report_to="none",
            seed=seed,
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=arguments, train_dataset=ReportDataset(items),
                          data_collator=PadCollator(pad_id))
        result = trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    return {"final_loss": result.training_loss, "losses": losses, "steps": total_steps}


# --------------------------------------------------------------------------
# Milestone: classify
# --------------------------------------------------------------------------

class Classifier:
    """Classifies reports with a fine-tuned model."""

    def __init__(self, model, tokenizer, labels: list[str], max_length: int = 96):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.labels = list(labels)
        self.tokens = [core.label_token(label) for label in self.labels]
        self.max_length = max_length
        self.device = next(model.parameters()).device
        ids = tokenizer.convert_tokens_to_ids(self.tokens)
        if any(i is None for i in ids):
            raise ValueError("the tokenizer does not contain the label tokens - was it saved with the model?")
        self.label_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id

    @classmethod
    def from_dir(cls, model_dir: str) -> "Classifier":
        """Load a classifier saved by `fine_tune`."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        meta = storage.read_meta(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model.to(device), tokenizer, meta["labels"], meta.get("max_length", 96))

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
        """For each report, the probability of each label (they add up to 1)."""
        results = []
        for start in range(0, len(texts), batch_size):
            prompts = [core.encode_prompt(self.tokenizer, text, self.max_length)
                       for text in texts[start : start + batch_size]]
            batch = core.pad_batch([{"input_ids": p, "labels": p} for p in prompts], self.pad_id)
            logits = self.model(
                input_ids=torch.tensor(batch["input_ids"], dtype=torch.long, device=self.device),
                attention_mask=torch.tensor(batch["attention_mask"], dtype=torch.long, device=self.device),
            ).logits
            last = torch.tensor([len(p) - 1 for p in prompts], device=self.device)      # last real token
            rows = logits[torch.arange(len(prompts), device=self.device), last]
            probabilities = torch.softmax(rows[:, self.label_ids], dim=-1)
            results += [dict(zip(self.labels, row)) for row in probabilities.tolist()]
        return results

    def classify(self, texts: list[str], min_confidence: float = 0.0, batch_size: int = 16) -> list[dict]:
        """Label, confidence and all scores for each report.

        Reports whose best score is below `min_confidence` get the label 'uncertain'.
        """
        output = []
        for text, scores in zip(texts, self.predict_proba(texts, batch_size)):
            label = max(scores, key=scores.get)
            confidence = scores[label]
            if confidence < min_confidence:
                label = "uncertain"
            output.append({"text": text, "label": label, "confidence": confidence, "scores": scores,
                           "token": core.label_token(label) if label != "uncertain" else "<uncertain>"})
        return output


def evaluate(classifier: Classifier, records: list[dict]) -> dict:
    """Predict every labelled record and return the true and predicted labels."""
    predictions = classifier.classify([r["text"] for r in records])
    return {"true": [r["label"] for r in records], "pred": [p["label"] for p in predictions]}


def fine_tune(model_name: str, records: list[dict], out_dir: str, epochs: int = 4, batch_size: int = 8,
              lr: float = 1e-4, max_length: int = 96, val_fraction: float = 0.1, seed: int = 1337,
              log=print) -> dict:
    """Add label tokens, train with Trainer, save the model and report validation accuracy."""
    labels = sorted({r["label"] for r in records})
    if len(labels) < 2:
        raise ValueError("need reports with at least 2 different labels")
    train_records, val_records = core.split_records(records, val_fraction, seed)

    model, tokenizer = load_base(model_name)
    ids = add_label_tokens(tokenizer, model, labels)
    log(f"added {len(labels)} label tokens: {', '.join(core.label_token(l) for l in labels)}")
    items = build_items(tokenizer, train_records, dict(zip(labels, ids)), max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    started = time.time()
    result = train_classifier(model, items, pad_id, epochs, batch_size, lr, seed)
    seconds = round(time.time() - started, 1)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    classifier = Classifier(model, tokenizer, labels, max_length)
    held_out = evaluate(classifier, val_records)
    val_accuracy = core.accuracy(held_out["true"], held_out["pred"]) if val_records else None
    meta = {
        "labels": labels, "tokens": classifier.tokens, "base_model": model_name, "max_length": max_length,
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed,
        "train_reports": len(train_records), "validation_reports": len(val_records),
        "validation_accuracy": val_accuracy, "final_loss": result["final_loss"], "seconds": seconds,
    }
    storage.write_meta(out_dir, meta)
    return meta
