# CreditLens-Risk — Lending Research Prototype

**Sector:** Finance  
**Task:** Binary classification: synthetic loan default

A synthetic loan application exercise using income, loan size, debt-to-income, credit history, delinquencies, employment duration, utilization, and age.

## Data and provenance

The offline demo uses a deterministic synthetic generator in `creditlens/data.py`. No public or private dataset is downloaded or redistributed. The generated values are illustrative and do not represent real people, organizations, or operational systems. Synthetic output is not evidence of real-world model performance.

## Method

- **Baseline:** majority-class prior classifier.
- **Stronger model:** histogram gradient-boosting classifier.
- **Split:** Stratified random holdout of independently generated applications.. The seed is fixed by default and configurable.
- **Evaluation:** accuracy, balanced accuracy, macro F1, confusion matrix, and (for binary tasks) ROC-AUC, average precision, precision, and recall at threshold 0.5.
- **Error analysis:** confusion matrix and misclassification count are emitted with the metrics.

Research/education only. These generated labels are not a validated credit policy. No lending decision should be made from this prototype; human review and jurisdiction-specific fairness evaluation are required. Age-band metrics are diagnostics only.

## Run locally

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m creditlens.train --seed 42 --samples 2400
    python -m creditlens.train --seed 42 --samples 2400 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` provide shortcuts. Python 3.9 or newer is required. No network access or data download is needed after dependencies are installed.

## Repository layout

- `creditlens/data.py` — validated domain-specific synthetic generator.
- `creditlens/split.py` — repeatable holdout split.
- `creditlens/model.py` — baseline, stronger model, metrics, and diagnostic summaries.
- `creditlens/train.py` — train/evaluate CLI.
- `tests/` — validation, split, model, and CLI tests.
- `.github/workflows/ci.yml` — automated test and CLI smoke run.

## Intended use

Educational and research prototype only. Do not use the synthetic benchmark to make decisions about real people or operations. Review the data assumptions, error patterns, and subgroup diagnostics before describing any findings.
