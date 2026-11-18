# Agricultural Report Classifier

A command-line tool that sorts short farmer reports and field observations into categories,
using GPT-2 fine-tuned to predict a special label token. Classify one report, or a whole file
of them, and get a confidence for each. It runs on a laptop CPU with no API keys.

> It sorts reports into categories so you can spot trends. It does **not** give agronomic
> advice, and a low-confidence result should be checked by a person.

## Categories

| Label (special token) | What it covers |
|---|---|
| `pest_alert` (`<pest_alert>`) | insects, mites, rodents, birds, and plant diseases such as rust or blight |
| `soil_condition` (`<soil_condition>`) | soil moisture, pH, compaction, salinity, erosion, nutrients |
| `weather_damage` (`<weather_damage>`) | hail, frost, flooding, wind, heat and drought damage |
| `crop_growth` (`<crop_growth>`) | germination, flowering, fruiting, progress and harvest readiness |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest

python -m app.cli train                                   # fine-tune GPT-2 (the first run downloads it)
python -m app.cli classify "Aphids are clustering under the maize leaves"
python -m app.cli classify --file my_reports.txt --output results.csv
python -m app.cli evaluate                                # accuracy on the handwritten test set
```

The datasets are already in `data/`. Re-create the training file with `python -m app.cli data`.

## Commands

| Command | What it does |
|---|---|
| `train` | Add the label tokens, fine-tune with the Hugging Face `Trainer`, and save the model to `models/classifier`. |
| `classify REPORT` | Classify one report and print the category token and its confidence. |
| `classify --file FILE` | Classify every report in a file, with a table and a summary count per category. |
| `evaluate` | Measure accuracy on labelled test reports, with per-label scores and a confusion matrix. |
| `data` | Write the synthetic training dataset. |

Useful options:

| Option | Meaning |
|---|---|
| `--scores` | Show the confidence for every category, not just the best. |
| `--min-confidence 0.6` | Label reports below this confidence as `<uncertain>`. |
| `--output results.csv` | Save results as `.csv` or `.jsonl`. |
| `--json` | Print one JSON object per report. |

Input files for `--file`: `.txt` with one report per line (blank lines and lines starting with
`#` are skipped), `.csv` with a `text` column, or `.jsonl` with a `text` field.

Output looks like this (the values depend on your trained model):

```
$ python -m app.cli classify "Aphids are clustering under the maize leaves" --scores
<pest_alert>  (xx.x%)
    pest_alert         xx.x%
    crop_growth        xx.x%
    ...

$ python -m app.cli classify --file my_reports.txt
#  category         confidence  report
-  ---------------  ----------  ------------------------------
1  <pest_alert>     xx%         Aphids are clustering under the maize leaves
2  <weather_damage> xx%         Hail hit the north field this morning...

Summary
  pest_alert           1  (50%)
  weather_damage       1  (50%)
```

## Milestones

| Milestone | Where |
|---|---|
| Define 3-4 categories and create sample text | `DEFAULT_LABELS`, `synthesize_dataset` in `app/core.py`; `data/reports.jsonl`, `data/test_handwritten.jsonl` |
| Extend the tokenizer and model with label tokens | `add_label_tokens` in `app/llm.py` (`add_special_tokens`, `resize_token_embeddings`) |
| Train with the Hugging Face Trainer | `train_classifier` and `fine_tune` in `app/llm.py` |
| CLI: classify one report | `classify` in `app/cli.py`, `Classifier` in `app/llm.py` |
| CLI: classify a file of reports | `classify --file` |
| Stretch: evaluate on a held-out test set | `evaluate`, `format_evaluation` in `app/core.py` |
| Stretch: confidence scores | printed by default; `--scores` shows all of them |
| Stretch: sub-categories | see below |

## How it works

1. **Label tokens.** Each label becomes one special token such as `<pest_alert>`. The tokenizer
   is extended with them and the model's embedding table is resized to make room.
2. **Training data.** Each report becomes `Report: <text>` and then `Category:`, followed by its
   label token. The loss is computed on the label token only, so the model is trained to do one
   thing: pick the right token after `Category:`. The `Trainer` fine-tunes the whole model.
3. **Classifying.** After `Category:`, the model scores every token. We read the scores for just
   the label tokens and turn them into probabilities. The best one is the category, and its
   probability is the confidence, so the tool can never answer with something that is not a
   label.

## Evaluation: use the handwritten test set

The training reports are generated from templates, so a model trained on them will do very well
on a test set made the same way. That number mostly shows the model learned the templates.
`data/test_handwritten.jsonl` has 40 reports (10 per category) written in different wording,
with no overlap with the training data. **Its accuracy is the number to report.**

`train` also prints a validation accuracy on held-out synthetic reports. Treat it as a sanity
check only.

To improve the result on real reports, add real labelled reports to a file in `data/` and
train with both: `python -m app.cli train --data data/reports.jsonl data/my_reports.jsonl`.

## Adding sub-categories

Labels come from your data, so nothing in the code is fixed to four categories. To split
`pest_alert`, label reports as `pest_alert_insect` and `pest_alert_disease` (letters, numbers
and underscores), put them in a file in `data/`, and retrain. Each label needs at least 5
reports, and more is better. `evaluate` then reports on every label, and its confusion matrix
shows which sub-categories get mixed up. Test files must only use labels the model has seen.

## Things to know

- **Training time.** Expect roughly 10 to 15 minutes for the defaults (4 epochs) on a laptop
  CPU. This is an estimate, so check on your machine. The first run downloads GPT-2 (about
  500 MB). Use `--model distilgpt2` for a faster, smaller model, or `--epochs 2` for a quick
  trial.
- **The model is about 500 MB**, so `models/` is ignored by git. Anyone can rebuild it with `train`.
- **Short reports.** Reports longer than `--max-length` tokens (default 96) are cut at the end.
- **Mixed reports.** A report about two things ("hail damage and aphids") gets one label.
  Check the confidence, or use `--min-confidence`.
- **Requirements.** The Hugging Face `Trainer` needs the `accelerate` package, which is in
  `requirements.txt`.

## Tests

```bash
pytest
```

The tests cover labels, file parsing, the datasets, token encoding, the evaluation metrics, every
CLI error path, the label-token and classifier code, and a full train, classify and evaluate run
on a tiny local model that needs no download. `.github/workflows/ci.yml` runs them on GitHub
after each push.

## Layout

```
app/core.py      labels, dataset parsing, synthetic data, token encoding, metrics (no PyTorch)
app/llm.py       label tokens, Trainer fine-tuning, the Classifier
app/storage.py   dataset and report files, results, saved model settings
app/cli.py       train / classify / evaluate / data
data/            reports.jsonl (synthetic training data), test_handwritten.jsonl
tests/           unit tests and end-to-end tests
```
