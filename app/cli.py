"""Command line interface for the Agricultural Report Classifier.

    python -m app.cli data                                   # write the starter dataset
    python -m app.cli train                                  # fine-tune GPT-2
    python -m app.cli classify "Aphids on the maize leaves"  # one report
    python -m app.cli classify --file reports.txt            # many reports, one per line
    python -m app.cli evaluate                               # accuracy on the handwritten test set

This tool sorts reports into categories. It does not give agronomic advice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import core, storage


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_data(args) -> int:
    records = core.synthesize_dataset(args.n, args.seed)
    storage.write_jsonl(args.out, records)
    print(f"wrote {len(records)} reports to {args.out}")
    print("reports per label:", core.label_counts(records))
    return 0


def cmd_train(args) -> int:
    records, problems = storage.read_records(args.data)
    if problems:
        print(f"skipped {len(problems)} unusable rows, for example: {problems[0]}")
    counts = core.label_counts(records)
    if len(counts) < 2:
        raise ValueError("training needs at least 2 different labels")
    if min(counts.values()) < 5:
        raise ValueError(f"every label needs at least 5 reports (found {counts})")
    print("reports per label:", counts)
    from . import llm

    meta = llm.fine_tune(args.model, records, args.out, args.epochs, args.batch_size, args.lr,
                         args.max_length, args.val_fraction, args.seed)
    print(f"\nsaved the model to {args.out} ({meta['seconds']}s)")
    if meta["validation_accuracy"] is not None:
        print(f"validation accuracy: {meta['validation_accuracy']:.1%} "
              f"on {meta['validation_reports']} held-out reports")
    print("Run `python -m app.cli evaluate` for the honest test on differently worded reports.")
    return 0


def _print_scores(row: dict) -> None:
    for label, score in sorted(row["scores"].items(), key=lambda item: -item[1]):
        print(f"    {label:<18} {score:.1%}")


def cmd_classify(args) -> int:
    texts = [" ".join(args.report)] if args.report else []
    if args.file:
        records, problems = storage.read_file(args.file, require_label=False)
        texts += [r["text"] for r in records]
        if problems:
            print(f"skipped {len(problems)} unusable lines, for example: {problems[0]}", file=sys.stderr)
    if not texts:
        raise ValueError("give a report to classify, or --file with one report per line")
    storage.read_meta(args.model_dir)                      # fail early if there is no trained model
    from . import llm

    rows = llm.Classifier.from_dir(args.model_dir).classify(texts, args.min_confidence)

    if args.json:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    elif len(rows) == 1 and not args.file:
        row = rows[0]
        print(f"{row['token']}  ({row['confidence']:.1%})")
        if args.scores:
            _print_scores(row)
    else:
        table = [[i, row["token"], f"{row['confidence']:.0%}", core.shorten(row["text"])]
                 for i, row in enumerate(rows, 1)]
        print(core.format_table(["#", "category", "confidence", "report"], table))
        print("\nSummary")
        for label, count, share in core.summarise_labels([row["label"] for row in rows]):
            print(f"  {label:<18} {count:>4}  ({share:.0%})")
        if args.scores:
            for i, row in enumerate(rows, 1):
                print(f"\n#{i} {core.shorten(row['text'], 50)}")
                _print_scores(row)
    if args.output:
        storage.write_results(args.output, rows)
        print(f"\nsaved results to {args.output}", file=sys.stderr)
    return 0


def cmd_evaluate(args) -> int:
    records, problems = storage.read_records(args.test)
    if problems:
        print(f"skipped {len(problems)} unusable rows, for example: {problems[0]}")
    if not records:
        raise ValueError("the test file has no usable labelled reports")
    meta = storage.read_meta(args.model_dir)
    unknown = sorted({r["label"] for r in records} - set(meta["labels"]))
    if unknown:
        raise ValueError(f"the test data has labels the model was not trained on: {', '.join(unknown)}")
    from . import llm

    result = llm.evaluate(llm.Classifier.from_dir(args.model_dir), records)
    report = core.format_evaluation("Test set", result["true"], result["pred"], meta["labels"])
    print(report)
    wrong = [(r, p) for r, p in zip(records, result["pred"]) if r["label"] != p]
    if wrong:
        print("\nMistakes")
        for record, predicted in wrong[:15]:
            print(f"  true {record['label']}, predicted {predicted}: {core.shorten(record['text'], 70)}")
    if args.out:
        storage.write_text(args.out, report + "\n")
        print(f"\nsaved the report to {args.out}")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Classify short agricultural reports (pest alert, soil condition, ...) "
                    "with a fine-tuned GPT-2 model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    model_dir = _env("MODEL_DIR", "models/classifier")

    data = sub.add_parser("data", help="write the synthetic starter dataset")
    data.add_argument("--n", type=int, default=320)
    data.add_argument("--seed", type=int, default=7)
    data.add_argument("--out", default="data/reports.jsonl")
    data.set_defaults(func=cmd_data)

    train = sub.add_parser("train", help="fine-tune GPT-2 on labelled reports")
    train.add_argument("--data", nargs="+", default=[_env("DATA_PATH", "data/reports.jsonl")],
                       help="one or more labelled .jsonl or .csv files")
    train.add_argument("--model", default=_env("BASE_MODEL", "gpt2"), help="base model, for example gpt2 or distilgpt2")
    train.add_argument("--out", default=model_dir, help="where to save the fine-tuned model")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--max-length", type=int, default=96, help="longest report, in tokens (longer ones are cut)")
    train.add_argument("--val-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=1337)
    train.set_defaults(func=cmd_train)

    classify = sub.add_parser("classify", help="classify one report, or every report in a file")
    classify.add_argument("report", nargs="*", help="the report text")
    classify.add_argument("--file", help="a .txt (one report per line), .csv (text column) or .jsonl file")
    classify.add_argument("--model-dir", default=model_dir)
    classify.add_argument("--scores", action="store_true", help="show the confidence for every category")
    classify.add_argument("--min-confidence", type=float, default=0.0,
                          help="label reports below this confidence as 'uncertain' (0 to 1)")
    classify.add_argument("--output", help="save results to a .csv or .jsonl file")
    classify.add_argument("--json", action="store_true", help="print one JSON object per report")
    classify.set_defaults(func=cmd_classify)

    evaluate = sub.add_parser("evaluate", help="measure accuracy on labelled test reports")
    evaluate.add_argument("--test", nargs="+", default=["data/test_handwritten.jsonl"])
    evaluate.add_argument("--model-dir", default=model_dir)
    evaluate.add_argument("--out", help="also save the report to this file")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv=None) -> int:
    storage.load_env()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ModuleNotFoundError as exc:
        if exc.name not in ("torch", "transformers", "accelerate"):
            raise
        print(f"error: {exc.name} is needed for this command - install it with: pip install -r requirements.txt",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
