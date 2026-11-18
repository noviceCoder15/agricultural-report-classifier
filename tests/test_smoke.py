import csv
import json

import pytest

from app import cli, core, storage


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---- the bundled datasets -----------------------------------------------------

def test_the_bundled_datasets_are_valid_and_balanced():
    train, train_problems = storage.read_file("data/reports.jsonl")
    test, test_problems = storage.read_file("data/test_handwritten.jsonl")
    assert train_problems == [] and test_problems == []
    assert core.label_counts(train) == {label: 80 for label in sorted(core.DEFAULT_LABELS)}
    assert core.label_counts(test) == {label: 10 for label in sorted(core.DEFAULT_LABELS)}


def test_no_test_report_appears_in_the_training_data():
    train, _ = storage.read_file("data/reports.jsonl")
    test, _ = storage.read_file("data/test_handwritten.jsonl")
    assert not {r["text"] for r in train} & {r["text"] for r in test}


def test_the_training_file_matches_the_generator():
    train, _ = storage.read_file("data/reports.jsonl")
    assert train == core.synthesize_dataset(320, seed=7)


# ---- storage --------------------------------------------------------------------

def test_reading_text_csv_and_jsonl_report_files(tmp_path):
    (tmp_path / "r.txt").write_text("Aphids on the maize.\nSoil is dry and cracking.\n")
    (tmp_path / "r.csv").write_text("\ufeffid,text\n1,\"Hail hit the north field, badly.\"\n", encoding="utf-8")
    (tmp_path / "r.jsonl").write_text(json.dumps({"text": "Maize is flowering."}) + "\n")
    assert [r["text"] for r in storage.read_file(str(tmp_path / "r.txt"), False)[0]] == [
        "Aphids on the maize.", "Soil is dry and cracking."]
    assert storage.read_file(str(tmp_path / "r.csv"), False)[0][0]["text"] == "Hail hit the north field, badly."
    assert storage.read_file(str(tmp_path / "r.jsonl"), False)[0][0]["text"] == "Maize is flowering."
    with pytest.raises(FileNotFoundError):
        storage.read_file(str(tmp_path / "missing.txt"))


def test_several_labelled_files_are_combined(tmp_path):
    for name, label in (("a.jsonl", "pest_alert"), ("b.jsonl", "soil_condition")):
        storage.write_jsonl(str(tmp_path / name), [{"text": "A report about something.", "label": label}])
    (tmp_path / "c.jsonl").write_text("oops\n")
    records, problems = storage.read_records([str(tmp_path / n) for n in ("a.jsonl", "b.jsonl", "c.jsonl")])
    assert [r["label"] for r in records] == ["pest_alert", "soil_condition"]
    assert problems == ["c.jsonl line 1: not valid JSON"]


def test_results_are_saved_as_csv_or_jsonl(tmp_path):
    rows = [{"text": "Aphids, again.", "label": "pest_alert", "confidence": 0.9,
             "scores": {"pest_alert": 0.9, "soil_condition": 0.1}}]
    storage.write_results(str(tmp_path / "out" / "r.csv"), rows)
    with open(tmp_path / "out" / "r.csv", newline="") as handle:
        table = list(csv.reader(handle))
    assert table[0] == ["text", "label", "confidence", "score_pest_alert", "score_soil_condition"]
    assert table[1] == ["Aphids, again.", "pest_alert", "0.9000", "0.9000", "0.1000"]
    storage.write_results(str(tmp_path / "r.jsonl"), rows)
    assert json.loads((tmp_path / "r.jsonl").read_text())["label"] == "pest_alert"
    with pytest.raises(ValueError):
        storage.write_results(str(tmp_path / "r.xlsx"), rows)


def test_model_settings_round_trip_and_a_missing_model_is_explained(tmp_path):
    storage.write_meta(str(tmp_path / "model"), {"labels": ["a", "b"]})
    assert storage.read_meta(str(tmp_path / "model")) == {"labels": ["a", "b"]}
    with pytest.raises(FileNotFoundError, match="train"):
        storage.read_meta(str(tmp_path / "nothing"))


# ---- command line (no model needed) ------------------------------------------------

def test_data_command_writes_a_dataset(tmp_path, capsys):
    out = tmp_path / "reports.jsonl"
    code, stdout, _ = run(capsys, "data", "--n", "40", "--out", str(out))
    assert code == 0 and "wrote 40 reports" in stdout
    assert len(storage.read_file(str(out))[0]) == 40


def test_train_checks_the_data_before_doing_anything_else(tmp_path, capsys):
    code, _, err = run(capsys, "train", "--data", str(tmp_path / "nope.jsonl"))
    assert code == 1 and "file not found" in err

    one_label = tmp_path / "one.jsonl"
    storage.write_jsonl(str(one_label), [{"text": f"Report number {i} here", "label": "pest_alert"} for i in range(9)])
    code, _, err = run(capsys, "train", "--data", str(one_label))
    assert code == 1 and "at least 2 different labels" in err

    few = tmp_path / "few.jsonl"
    storage.write_jsonl(str(few), [{"text": f"Report number {i} here", "label": "a" if i < 8 else "b"} for i in range(10)])
    code, _, err = run(capsys, "train", "--data", str(few))
    assert code == 1 and "at least 5 reports" in err


def test_classify_needs_a_report_and_a_trained_model(tmp_path, capsys):
    code, _, err = run(capsys, "classify")
    assert code == 1 and "give a report" in err
    code, _, err = run(capsys, "classify", "Aphids on the maize", "--model-dir", str(tmp_path / "none"))
    assert code == 1 and "no trained model" in err


def test_classify_reports_unreadable_input_files(tmp_path, capsys):
    code, _, err = run(capsys, "classify", "--file", str(tmp_path / "nope.txt"), "--model-dir", str(tmp_path))
    assert code == 1 and "file not found" in err


def test_evaluate_rejects_labels_the_model_does_not_know(tmp_path, capsys):
    storage.write_meta(str(tmp_path / "model"), {"labels": ["pest_alert", "soil_condition"]})
    code, _, err = run(capsys, "evaluate", "--model-dir", str(tmp_path / "model"))
    assert code == 1 and "not trained on" in err and "weather_damage" in err
    code, _, err = run(capsys, "evaluate", "--model-dir", str(tmp_path / "missing"))
    assert code == 1 and "no trained model" in err


# ---- the whole pipeline, offline, on a tiny local model ------------------------------

def make_tiny_model_folder(folder):
    """A tiny random GPT-2 with a byte-level tokenizer, saved locally so no download is needed."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {char: i for i, char in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    vocab["<|endoftext|>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>").save_pretrained(folder)
    config = GPT2Config(vocab_size=len(vocab) + 8, n_positions=128, n_embd=32, n_layer=2, n_head=2,
                        bos_token_id=vocab["<|endoftext|>"], eos_token_id=vocab["<|endoftext|>"])
    GPT2LMHeadModel(config).save_pretrained(folder)
    return str(folder)


def test_train_classify_and_evaluate_end_to_end(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    base = make_tiny_model_folder(tmp_path / "tiny")
    data = tmp_path / "train.jsonl"
    storage.write_jsonl(str(data), core.synthesize_dataset(40, seed=1))
    out = str(tmp_path / "model")

    code, stdout, _ = run(capsys, "train", "--data", str(data), "--model", base, "--out", out,
                          "--epochs", "1", "--batch-size", "8", "--max-length", "64", "--seed", "1")
    assert code == 0 and "added 4 label tokens" in stdout and "validation accuracy" in stdout
    assert storage.read_meta(out)["labels"] == sorted(core.DEFAULT_LABELS)

    code, stdout, _ = run(capsys, "classify", "Aphids on the maize leaves", "--model-dir", out, "--scores")
    assert code == 0 and stdout.startswith("<") and "%" in stdout
    for label in core.DEFAULT_LABELS:
        assert label in stdout

    reports = tmp_path / "reports.txt"
    reports.write_text("Aphids on the maize leaves.\nHail hit the north field.\nSoil is dry.\n")
    results = tmp_path / "results.csv"
    code, stdout, _ = run(capsys, "classify", "--file", str(reports), "--model-dir", out, "--output", str(results))
    assert code == 0 and "Summary" in stdout
    assert len(list(csv.reader(open(results, newline="")))) == 4                # header + 3 reports

    code, stdout, _ = run(capsys, "classify", "Hail hit the north field.", "--model-dir", out, "--json")
    row = json.loads(stdout.strip().splitlines()[0])
    assert set(row["scores"]) == set(core.DEFAULT_LABELS) and row["label"] in core.DEFAULT_LABELS

    code, stdout, _ = run(capsys, "classify", "Hail hit the north field.", "--model-dir", out, "--min-confidence", "1.01")
    assert "<uncertain>" in stdout

    code, stdout, _ = run(capsys, "evaluate", "--test", str(data), "--model-dir", out)
    assert code == 0 and "accuracy" in stdout and "Confusion matrix" in stdout