"""Labels, prompts, datasets and metrics for the Agricultural Report Classifier.

Pure Python, standard library only, so it can be tested without PyTorch.

How classification works: each report becomes the prompt

    Report: <text>
    Category:

and the model is trained to predict ONE special token next, such as <pest_alert>.
To classify, we read the model's scores for just the label tokens and pick the
highest, which also gives a confidence.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
from collections import Counter, defaultdict

PREFIX = "Report:"
SUFFIX = "\nCategory:"
LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
MIN_TEXT_CHARS, MAX_TEXT_CHARS = 5, 1000
DEFAULT_LABELS = ("pest_alert", "soil_condition", "weather_damage", "crop_growth")

# --------------------------------------------------------------------------
# Labels and label tokens
# --------------------------------------------------------------------------


def normalise_label(raw) -> str:
    """'Pest Alert', 'pest-alert' and '<pest_alert>' all become 'pest_alert'."""
    label = str(raw or "").strip().strip("<>").strip().lower()
    label = re.sub(r"[\s-]+", "_", label)
    if not LABEL_PATTERN.match(label):
        raise ValueError(f"'{raw}' is not a usable label - use letters, numbers and underscores, like pest_alert")
    return label


def label_token(label: str) -> str:
    """The special token for a label: 'pest_alert' -> '<pest_alert>'."""
    return f"<{normalise_label(label)}>"


# --------------------------------------------------------------------------
# Reading datasets
# --------------------------------------------------------------------------


def normalise_record(record: dict, require_label: bool = True):
    """Turn a raw row into {'text', 'label'?, 'source'?}. Returns (record, []) or (None, problems)."""
    rec = {str(k).strip().lower(): v for k, v in record.items() if k is not None}
    problems: list[str] = []
    text = " ".join(str(rec.get("text") or rec.get("report") or "").split())
    if len(text) < MIN_TEXT_CHARS:
        problems.append("report text is missing or too short")
    elif len(text) > MAX_TEXT_CHARS:
        problems.append("report text is too long")

    label = None
    raw_label = rec.get("label") or rec.get("category")
    if raw_label not in (None, ""):
        try:
            label = normalise_label(raw_label)
        except ValueError as exc:
            problems.append(str(exc))
    elif require_label:
        problems.append("missing label")

    if problems:
        return None, problems
    result = {"text": text}
    if label:
        result["label"] = label
    if rec.get("source"):
        result["source"] = str(rec["source"])
    return result, []


def parse_records(text: str, kind: str = "jsonl", require_label: bool = True):
    """Parse JSONL, CSV or plain-text (one report per line) content.

    Returns (records, problems); unusable rows are skipped with a reason.
    Plain text has no labels, so it is only allowed when require_label is False.
    """
    kind = kind.lower().lstrip(".")
    rows: list[tuple[str, dict]] = []
    problems: list[str] = []
    if kind in ("jsonl", "ndjson"):
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {number}: not valid JSON")
                continue
            if not isinstance(record, dict):
                problems.append(f"line {number}: expected a JSON object")
                continue
            rows.append((f"line {number}", record))
    elif kind == "csv":
        for number, row in enumerate(csv.DictReader(io.StringIO(text)), 2):
            rows.append((f"row {number}", row))
    elif kind in ("txt", ""):
        if require_label:
            raise ValueError("a plain text file has no labels - use .jsonl or .csv with a label column")
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append((f"line {number}", {"text": line}))
    else:
        raise ValueError(f"unsupported file type '{kind}' - use .jsonl, .csv or .txt")

    records = []
    for label, row in rows:
        record, issues = normalise_record(row, require_label)
        if record:
            records.append(record)
        else:
            problems.extend(f"{label}: {issue}" for issue in issues)
    return records, problems


def split_records(records: list[dict], val_fraction: float = 0.1, seed: int = 0):
    """Hold out a share of every label (reproducibly) for validation."""
    if len(records) < 2:
        raise ValueError("need at least 2 labelled reports")
    by_label: dict[str, list] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        n_val = min(len(group) - 1, max(1, n_val)) if len(group) >= 2 else 0
        val += group[:n_val]
        train += group[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def label_counts(records: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(r["label"] for r in records).items()))


# --------------------------------------------------------------------------
# A synthetic starter dataset
# --------------------------------------------------------------------------
# Templates for four report types, filled with random crops, places and numbers.
# Neutral openers and closers are shared by all labels, so they carry no signal.
# Synthetic text is far more regular than real reports, so test on real wording
# too (data/test_handwritten.jsonl).

CROPS = ["maize", "wheat", "rice", "cotton", "sorghum", "millet", "barley", "sugarcane", "cassava", "soybean"]
PLACES = ["the north field", "block C", "the lower paddock", "plot 7", "the river terrace", "the east side",
          "row 12", "the hillside plot", "the front field", "field B", "the back acre", "the school field"]
TIMES = ["this morning", "yesterday", "on Tuesday", "two days ago", "this week", "on Friday afternoon",
         "earlier today", "last week"]
OPENERS = ["Field note:", "Update:", "Report:", "Quick note:", "Morning round:", "FYI:"]
CLOSERS = ["Will check again tomorrow.", "Photos attached.", "Please advise.", "No action taken yet.",
           "Logged by the field supervisor.", "Neighbours are seeing the same.", "Will update this evening."]

TEMPLATES = {
    "pest_alert": [
        "Found {num} armyworm larvae per plant on the {crop} in {place} {time}.",
        "Aphids are clustering under the leaves of the {crop} in {place}.",
        "A locust swarm was spotted moving across {place} {time}.",
        "Leaf rust spots are spreading on the {crop} in {place}.",
        "Rats have been chewing stored {crop} near {place}.",
        "Whiteflies cover the underside of the {crop} leaves in {place}.",
        "Fall armyworm damage on about {pct}% of the {crop} plants in {place}.",
        "Birds are pecking at the ripening {crop} in {place}.",
        "Stem borer holes are visible on the {crop} stalks in {place}.",
        "Fungal blight lesions have appeared on the {crop} leaves in {place}.",
        "Spider mites and fine webbing on the {crop} in {place}.",
        "Caterpillars ate holes through the {crop} leaves in {place} {time}.",
    ],
    "soil_condition": [
        "The soil in {place} is very dry and cracking.",
        "Soil pH in {place} came back at {ph}, too acidic for {crop}.",
        "The soil is badly compacted in {place}, roots cannot go deep.",
        "A white salt crust has formed on the soil surface in {place}.",
        "Soil moisture in {place} is only {pct}% at 20 cm depth.",
        "Topsoil is eroding along the slope in {place}.",
        "Lab results show low nitrogen in the soil of {place}.",
        "The ground in {place} is waterlogged and heavy clay sticks to the hoe.",
        "Organic matter is low in {place}, the soil looks pale and sandy.",
        "The soil in {place} is hard as brick, the plough could not break it.",
        "A soil test confirmed phosphorus deficiency in {place}.",
        "Earthworms are plentiful and the soil in {place} is dark and crumbly.",
    ],
    "weather_damage": [
        "Hail hit {place} {time} and shredded the {crop} leaves.",
        "Frost last night burned the young {crop} in {place}.",
        "Strong winds knocked down about {pct}% of the {crop} in {place}.",
        "Flooding after the storm covered {place}, the {crop} is under water.",
        "A heatwave scorched the {crop} in {place}, leaves are curling and browning.",
        "Heavy rain {time} broke the {crop} stalks in {place}.",
        "A late frost killed the {crop} blossoms in {place}.",
        "No rain for {weeks} weeks and the drought has stunted the {crop} in {place}.",
        "Cyclone winds tore the plastic cover off the greenhouse at {place}.",
        "Hailstones the size of marbles dented the {crop} pods in {place}.",
        "A storm {time} flattened the {crop} across {place}.",
        "Scorching heat {time} has withered the {crop} in {place}.",
    ],
    "crop_growth": [
        "The {crop} in {place} has reached the flowering stage.",
        "Germination is good, about {pct}% of the {crop} seeds have sprouted in {place}.",
        "The {crop} plants in {place} are about {num} cm tall and look healthy.",
        "Fruit set has started on the {crop} in {place}.",
        "The {crop} in {place} is ready for harvest, the grain is hard and golden.",
        "Tillering is going well on the {crop} in {place}, plants look strong.",
        "{crop} seedlings were transplanted in {place} and are taking root.",
        "Pods are filling nicely on the {crop} in {place}.",
        "Expect a good yield from the {crop} in {place}, the heads are well formed.",
        "The canopy has closed over the rows in {place}, {crop} growth is on schedule.",
        "The {crop} in {place} is two weeks behind schedule and growing slowly.",
        "Leaves on the {crop} in {place} are a deep green and growing fast.",
    ],
}


def _synthetic_record(rng: random.Random, label: str) -> dict:
    values = {
        "crop": rng.choice(CROPS), "place": rng.choice(PLACES), "time": rng.choice(TIMES),
        "num": rng.randint(2, 40), "pct": rng.randrange(5, 95, 5), "weeks": rng.randint(3, 9),
        "ph": f"{rng.randint(45, 54) / 10:.1f}",
    }
    text = rng.choice(TEMPLATES[label]).format(**values)
    if rng.random() < 0.3:
        text = f"{rng.choice(OPENERS)} {text}"
    if rng.random() < 0.4:
        text = f"{text} {rng.choice(CLOSERS)}"
    return {"text": text, "label": label, "source": "synthetic"}


def synthesize_dataset(n: int = 320, seed: int = 7) -> list[dict]:
    """A reproducible, label-balanced starter dataset with no duplicate reports."""
    rng = random.Random(seed)
    labels = list(DEFAULT_LABELS)
    records, seen = [], set()
    for i in range(n):
        label = labels[i % len(labels)]
        for _ in range(50):                            # retry if the random fill repeats an earlier report
            record = _synthetic_record(rng, label)
            if record["text"] not in seen:
                break
        seen.add(record["text"])
        records.append(record)
    return records


# --------------------------------------------------------------------------
# Turning reports into token ids (works with any tokenizer that has encode())
# --------------------------------------------------------------------------


def encode_prompt(tokenizer, text: str, max_length: int = 96) -> list[int]:
    """Token ids for 'Report: <text>\\nCategory:'. Long reports are cut, never the framing."""
    prefix = list(tokenizer.encode(PREFIX))
    suffix = list(tokenizer.encode(SUFFIX))
    body = list(tokenizer.encode(" " + " ".join(text.split())))
    room = max_length - len(prefix) - len(suffix) - 1          # one slot is kept for the label token
    if room < 1:
        raise ValueError("max_length is too small for a report")
    return prefix + body[:room] + suffix


def encode_example(tokenizer, text: str, label_id: int, max_length: int = 96) -> dict:
    """Prompt ids plus the label token. The loss is only on the label token."""
    prompt = encode_prompt(tokenizer, text, max_length)
    return {"input_ids": prompt + [label_id], "labels": [-100] * len(prompt) + [label_id]}


def pad_batch(items: list[dict], pad_id: int) -> dict:
    """Right-pad tokenized examples to the same length."""
    longest = max(len(item["input_ids"]) for item in items)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in items:
        pad = longest - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
        batch["attention_mask"].append([1] * len(item["input_ids"]) + [0] * pad)
        batch["labels"].append(item["labels"] + [-100] * pad)
    return batch


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def accuracy(true: list[str], pred: list[str]) -> float:
    if not true:
        raise ValueError("nothing to evaluate")
    return sum(t == p for t, p in zip(true, pred)) / len(true)


def confusion_matrix(true: list[str], pred: list[str], labels: list[str]) -> list[list[int]]:
    """Rows are the true labels, columns the predicted ones, in `labels` order."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(true, pred):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix


def label_metrics(true: list[str], pred: list[str], labels: list[str]) -> dict[str, dict]:
    """Precision, recall, F1 and support for each label."""
    metrics = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true, pred))
        predicted = sum(p == label for p in pred)
        actual = sum(t == label for t in true)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return metrics


def format_table(headers, rows) -> str:
    widths = [max([len(str(h))] + [len(str(row[i])) for row in rows]) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows]
    return "\n".join([line, rule] + body)


def format_evaluation(title: str, true: list[str], pred: list[str], labels: list[str]) -> str:
    """Accuracy, per-label scores and a confusion matrix as plain text."""
    metrics = label_metrics(true, pred, labels)
    rows = [[label, f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}", m["support"]]
            for label, m in metrics.items()]
    matrix = confusion_matrix(true, pred, labels)
    confusion_rows = [[label] + counts for label, counts in zip(labels, matrix)]
    return "\n".join([
        f"{title}: accuracy {accuracy(true, pred):.1%} on {len(true)} reports", "",
        format_table(["label", "precision", "recall", "f1", "reports"], rows), "",
        "Confusion matrix (rows = true label, columns = predicted)",
        format_table(["true \\ predicted"] + labels, confusion_rows),
    ])


def shorten(text: str, width: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3].rstrip() + "..."


def summarise_labels(labels: list[str]) -> list[tuple[str, int, float]]:
    """(label, count, share) for a list of predictions, most common first."""
    counts = Counter(labels)
    total = max(1, len(labels))
    return [(label, count, count / total) for label, count in counts.most_common()]
