"""File I/O: .env, datasets, report files, results and the saved model's settings."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from . import core

META_FILE = "classifier.json"


def load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a .env file into os.environ (existing vars win)."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# --------------------------------------------------------------------------
# Reading datasets and report files
# --------------------------------------------------------------------------

def read_file(path: str, require_label: bool = True):
    """Read one .jsonl, .csv or .txt file. Returns (records, problems)."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return core.parse_records(file.read_text(encoding="utf-8-sig"), file.suffix, require_label)


def read_records(paths, require_label: bool = True):
    """Read and combine one or more labelled files. Returns (records, problems)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records, problems = [], []
    for path in paths:
        found, issues = read_file(str(path), require_label)
        records += found
        problems += [f"{Path(path).name} {issue}" for issue in issues]
    return records, problems


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_jsonl(path: str, records: list) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def write_text(path: str, text: str) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")


def write_results(path: str, rows: list[dict]) -> None:
    """Save classification results as .csv or .jsonl.

    Each row: {"text", "label", "confidence", "scores": {label: probability}}.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    kind = file.suffix.lower()
    if kind == ".jsonl":
        write_jsonl(path, rows)
    elif kind == ".csv":
        labels = list(rows[0]["scores"]) if rows else []
        with file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "label", "confidence"] + [f"score_{label}" for label in labels])
            for row in rows:
                writer.writerow([row["text"], row["label"], f"{row['confidence']:.4f}"]
                                + [f"{row['scores'][label]:.4f}" for label in labels])
    else:
        raise ValueError(f"cannot write '{file.suffix}' results - use a .csv or .jsonl file name")


# --------------------------------------------------------------------------
# The saved model's settings
# --------------------------------------------------------------------------

def write_meta(model_dir: str, meta: dict) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(model_dir: str) -> dict:
    """Labels and settings saved next to the model. Raises a clear error if there is no model."""
    meta_file = Path(model_dir) / META_FILE
    if not meta_file.exists():
        raise FileNotFoundError(f"no trained model at {model_dir} - run `python -m app.cli train` first")
    return json.loads(meta_file.read_text(encoding="utf-8"))
