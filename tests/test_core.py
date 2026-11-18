import json

import pytest

from app import core


class FakeTokenizer:
    """Character-level stand-in for a Hugging Face tokenizer (ids 2-61), with label-token support."""

    eos_token_id = 1
    pad_token_id = 1
    base_size = 64

    def __init__(self):
        self.added = []

    def __len__(self):
        return self.base_size + len(self.added)

    def encode(self, text):
        return [2 + (ord(c) % 60) for c in text]

    def add_special_tokens(self, mapping):
        for token in mapping["additional_special_tokens"]:
            if token not in self.added:
                self.added.append(token)

    def convert_tokens_to_ids(self, tokens):
        return [self.base_size + self.added.index(t) if t in self.added else None for t in tokens]


# ---- labels ----------------------------------------------------------------

def test_labels_are_normalised():
    for raw in ("pest_alert", "Pest Alert", "pest-alert", "<PEST_ALERT>", " pest_alert "):
        assert core.normalise_label(raw) == "pest_alert"
    assert core.label_token("Soil Condition") == "<soil_condition>"


def test_unusable_labels_are_rejected():
    for bad in ("", "1abc", "pest!", "   ", None):
        with pytest.raises(ValueError):
            core.normalise_label(bad)


# ---- reading datasets --------------------------------------------------------

def test_jsonl_records_are_cleaned_and_bad_rows_explained():
    text = "\n".join([
        json.dumps({"text": "  Aphids   on the maize. ", "label": "Pest Alert", "source": "test"}),
        "not json",
        json.dumps(["a list"]),
        json.dumps({"text": "Soil is dry.", "label": ""}),
        json.dumps({"text": "x", "label": "pest_alert"}),
        json.dumps({"text": "Hail hit the north field.", "label": "bad label!"}),
        "",
        json.dumps({"report": "Maize is flowering.", "category": "crop growth"}),
    ])
    records, problems = core.parse_records(text, "jsonl")
    assert records == [
        {"text": "Aphids on the maize.", "label": "pest_alert", "source": "test"},
        {"text": "Maize is flowering.", "label": "crop_growth"},
    ]
    joined = " | ".join(problems)
    for expected in ("line 2: not valid JSON", "line 3: expected a JSON object", "line 4: missing label",
                     "line 5: report text is missing or too short", "line 6:"):
        assert expected in joined


def test_csv_records_use_a_header_row():
    text = "Text,Label\n\"Aphids on the maize, again\",pest_alert\n,soil_condition\n"
    records, problems = core.parse_records(text, "csv")
    assert records == [{"text": "Aphids on the maize, again", "label": "pest_alert"}]
    assert problems and problems[0].startswith("row 3")


def test_plain_text_has_one_report_per_line_and_no_labels():
    text = "# a comment\nAphids on the maize.\n\n   \nSoil is dry and cracking.\n"
    records, problems = core.parse_records(text, ".txt", require_label=False)
    assert [r["text"] for r in records] == ["Aphids on the maize.", "Soil is dry and cracking."]
    assert problems == [] and all("label" not in r for r in records)
    with pytest.raises(ValueError):
        core.parse_records(text, "txt", require_label=True)


def test_unknown_file_type_is_rejected():
    with pytest.raises(ValueError):
        core.parse_records("{}", "xlsx")


def test_split_keeps_every_label_in_both_parts():
    records = [{"text": f"report {i} about things", "label": label}
               for label in ("a", "b", "c") for i in range(20)]
    train, val = core.split_records(records, 0.1, seed=4)
    assert len(val) == 6 and len(train) == 54
    assert core.label_counts(val) == {"a": 2, "b": 2, "c": 2}
    assert not [r for r in train if r in val]
    assert core.split_records(records, 0.1, seed=4) == (train, val)
    with pytest.raises(ValueError):
        core.split_records(records[:1])


# ---- the synthetic dataset ---------------------------------------------------

def test_synthetic_dataset_is_balanced_unique_and_reproducible():
    data = core.synthesize_dataset(80, seed=3)
    assert data == core.synthesize_dataset(80, seed=3)
    assert data != core.synthesize_dataset(80, seed=4)
    assert core.label_counts(data) == {label: 20 for label in sorted(core.DEFAULT_LABELS)}
    assert len({r["text"] for r in data}) == 80
    for record in data:
        assert core.normalise_record(record)[0] == record


# ---- turning reports into token ids -------------------------------------------

def test_prompt_is_framed_by_report_and_category():
    tok = FakeTokenizer()
    ids = core.encode_prompt(tok, "Aphids on the maize.", max_length=96)
    assert ids[: len("Report:")] == tok.encode("Report:")
    assert ids[-len("\nCategory:"):] == tok.encode("\nCategory:")
    assert ids == core.encode_prompt(tok, "Aphids   on  the maize.", max_length=96)      # spacing is normalised


def test_long_reports_are_cut_but_the_framing_is_kept():
    tok = FakeTokenizer()
    ids = core.encode_prompt(tok, "word " * 200, max_length=50)
    assert len(ids) == 49                                    # one slot stays free for the label token
    assert ids[-len("\nCategory:"):] == tok.encode("\nCategory:")
    with pytest.raises(ValueError):
        core.encode_prompt(tok, "text", max_length=10)


def test_only_the_label_token_is_trained_on():
    tok = FakeTokenizer()
    example = core.encode_example(tok, "Aphids on the maize.", label_id=65, max_length=96)
    prompt = core.encode_prompt(tok, "Aphids on the maize.", 96)
    assert example["input_ids"] == prompt + [65]
    assert example["labels"] == [-100] * len(prompt) + [65]


def test_pad_batch_pads_inputs_masks_and_labels():
    batch = core.pad_batch([{"input_ids": [5, 6, 7], "labels": [-100, -100, 7]},
                            {"input_ids": [8], "labels": [8]}], pad_id=1)
    assert batch["input_ids"] == [[5, 6, 7], [8, 1, 1]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"] == [[-100, -100, 7], [8, -100, -100]]


# ---- evaluation ---------------------------------------------------------------

TRUE = ["a", "a", "b", "b", "b"]
PRED = ["a", "b", "b", "b", "a"]


def test_accuracy_and_confusion_matrix():
    assert core.accuracy(TRUE, PRED) == pytest.approx(0.6)
    assert core.confusion_matrix(TRUE, PRED, ["a", "b"]) == [[1, 1], [1, 2]]
    with pytest.raises(ValueError):
        core.accuracy([], [])


def test_precision_recall_and_f1_per_label():
    metrics = core.label_metrics(TRUE, PRED, ["a", "b"])
    assert metrics["a"]["precision"] == pytest.approx(0.5) and metrics["a"]["recall"] == pytest.approx(0.5)
    assert metrics["b"]["precision"] == pytest.approx(2 / 3) and metrics["b"]["recall"] == pytest.approx(2 / 3)
    assert metrics["a"]["support"] == 2 and metrics["b"]["support"] == 3
    assert core.label_metrics(["a"], ["a"], ["a", "z"])["z"]["f1"] == 0.0        # a label with no reports


def test_evaluation_report_has_accuracy_and_matrix():
    report = core.format_evaluation("Test set", TRUE, PRED, ["a", "b"])
    assert "accuracy 60.0% on 5 reports" in report
    assert "Confusion matrix" in report and "precision" in report


def test_display_helpers():
    assert core.shorten("a " * 100, 20).endswith("...") and len(core.shorten("a " * 100, 20)) <= 20
    assert core.shorten("short text") == "short text"
    assert core.summarise_labels(["a", "b", "a"]) == [("a", 2, pytest.approx(2 / 3)), ("b", 1, pytest.approx(1 / 3))]


# ---- the model (needs PyTorch and transformers) --------------------------------------

def torch_llm():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from app import llm

    return torch, llm


def tiny_gpt2(vocab_size=64):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(vocab_size=vocab_size, n_positions=128, n_embd=32, n_layer=2, n_head=2,
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(config).eval()


LABELS = ["crop_growth", "pest_alert", "soil_condition", "weather_damage"]


def tiny_classifier(llm):
    model, tokenizer = tiny_gpt2(), FakeTokenizer()
    llm.add_label_tokens(tokenizer, model, LABELS)
    return llm.Classifier(model, tokenizer, LABELS, max_length=96)


def test_label_tokens_are_added_to_the_tokenizer_and_the_model():
    _, llm = torch_llm()
    model, tokenizer = tiny_gpt2(), FakeTokenizer()
    ids = llm.add_label_tokens(tokenizer, model, LABELS)
    assert ids == [64, 65, 66, 67]
    assert tokenizer.added == ["<crop_growth>", "<pest_alert>", "<soil_condition>", "<weather_damage>"]
    assert model.get_input_embeddings().weight.shape[0] == 68
    assert model.get_output_embeddings().weight.shape[0] == 68


def test_scores_are_probabilities_over_the_labels():
    _, llm = torch_llm()
    classifier = tiny_classifier(llm)
    scores = classifier.predict_proba(["Aphids on the maize.", "Hail hit the north field."])
    for row in scores:
        assert list(row) == LABELS
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-5)


def test_a_report_scores_the_same_alone_or_in_a_padded_batch():
    _, llm = torch_llm()
    classifier = tiny_classifier(llm)
    texts = ["Aphids.", "A much longer report about leaf rust spreading across the whole north field."]
    together = classifier.predict_proba(texts)
    for text, row in zip(texts, together):
        alone = classifier.predict_proba([text])[0]
        assert all(abs(alone[label] - row[label]) < 1e-4 for label in LABELS)


def test_low_confidence_reports_can_be_marked_uncertain():
    _, llm = torch_llm()
    classifier = tiny_classifier(llm)
    first = classifier.classify(["Aphids on the maize."])[0]
    assert first["label"] in LABELS and first["token"] == f"<{first['label']}>"
    assert 0.25 <= first["confidence"] <= 1.0
    assert classifier.classify(["Aphids on the maize."], min_confidence=1.01)[0]["label"] == "uncertain"


def test_trainer_runs_on_label_only_examples():
    torch, llm = torch_llm()
    pytest.importorskip("accelerate")
    model, tokenizer = tiny_gpt2(), FakeTokenizer()
    ids = dict(zip(LABELS, llm.add_label_tokens(tokenizer, model, LABELS)))
    records = core.synthesize_dataset(16, seed=1)
    items = llm.build_items(tokenizer, records, ids, max_length=64)
    before = model.get_input_embeddings().weight.detach().clone()
    result = llm.train_classifier(model, items, pad_id=1, epochs=1, batch_size=4, lr=1e-3, seed=1)
    assert result["steps"] == 4 and torch.isfinite(torch.tensor(result["final_loss"]))
    assert not torch.equal(model.get_input_embeddings().weight.detach().cpu(), before.cpu())
