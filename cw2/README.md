# CW2: Hybrid SOC alert-triage prototype (TeleComNet Ltd)

7CS537 AI in Cybersecurity, Coursework 2. Student number: 100810453.

This folder holds the CW2 artefact: an academic prototype that helps a Security Operations Centre (SOC) decide which network flows to review first. It combines two models:

- **XGBoost** (supervised) learns attack types it has already seen.
- **An autoencoder** (unsupervised, trained only on normal Monday traffic) flags flows that look unlike normal traffic, which can include attack families no model was trained on.

Alerts are ranked into a daily queue sized to what the team can review, and each alert comes with an explanation. A Streamlit dashboard shows this to an analyst.

CW1 (the notebooks in the repository root) is unchanged. CW2 reuses only CW1's cleaned dataset and feature list.

## How it is evaluated

Unlike CW1's random split, CW2 tests models on **later days than they were trained on**, which is closer to how a SOC would deploy them:

| Experiment | Train and validate on | Test on |
|---|---|---|
| E1 (main test) | Monday to Thursday | Friday |
| E2 Wednesday | Monday and Tuesday | Wednesday |
| E2 Thursday | Monday to Wednesday | Thursday |
| Random (reference) | CW1's 70/15/15 split, seed 42 | CW1's test set |
| E3 held-out attack family | Random split with one family removed from training | That family |

In CICIDS2017 every attack type occurs on only one day, so each time-ordered test contains only attack families the supervised model has never seen. The autoencoder's alert threshold is set on benign traffic only (99th percentile, a 1% false alarm target), and its real false alarm rate is always reported.

## Contents

| File | What it does | Main outputs |
|---|---|---|
| `03_data_and_splits.ipynb` | Builds all splits, checks the random split matches CW1, counts test flows identical to training flows | `data/03_splits.parquet`, `outputs/03_*` |
| `04_xgboost_chronological.ipynb` | Retrains XGBoost (CW1 settings) on each split | `models/04_xgb_*.json`, `data/04_xgb_test_scores.parquet`, `outputs/04_*` |
| `05_autoencoder.ipynb` | Trains the autoencoder and an Isolation Forest on benign Monday traffic; sets thresholds; compares day by day | `models/05_*`, `data/05_anomaly_scores.parquet`, `outputs/05_*` |
| `05b_held_out_family.ipynb` | E3: each attack family held out of XGBoost training in turn | `outputs/05b_*` |
| `06_hybrid_triage.ipynb` | Priority tiers, capacity-limited daily queues, threshold sensitivity, per-alert explanations | `data/06_alert_queue.parquet`, `outputs/06_*` |
| `app/triage_dashboard.py` | Streamlit analyst dashboard | (reads the saved models and scores) |

`data/` and `models/` are not pushed to GitHub (see `.gitignore`); running the notebooks recreates them.

## How to run (macOS)

From the repository root:

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r cw2/requirements.txt
```

1. Run CW1 Notebook 01 first if `data/cicids2017_clean.parquet` does not exist (it needs the eight CICIDS2017 CSV files in `data/`).
2. Run the CW2 notebooks in order, each from a fresh kernel:

```zsh
cd cw2
for nb in 03_data_and_splits 04_xgboost_chronological 05_autoencoder 05b_held_out_family 06_hybrid_triage; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" --ExecutePreprocessor.timeout=-1
done
cd ..
```

3. Start the dashboard:

```zsh
streamlit run cw2/app/triage_dashboard.py
```

Each notebook checks a fingerprint of the data (`cb8a0f0b8bd9cf88`) so results are not mixed across different versions of the cleaned dataset.

**macOS note.** PyTorch and XGBoost load different copies of the OpenMP library, and running both in one Python process crashed (segmentation fault). For that reason Notebook 05 does not use XGBoost, Notebook 05b does not use PyTorch, and Notebook 06 and the dashboard run the trained autoencoder in NumPy (the weights are converted once, in a separate process). Notebook 06 checks that the NumPy version gives the same errors as PyTorch.

## Dashboard

The sidebar lets the analyst choose:

- the test day (Friday, Wednesday or Thursday)
- how alerts are ranked: hybrid tiers, XGBoost score, or autoencoder error
- how many alerts the team can review that day (100 to 5,000)
- the autoencoder's false alarm target (0.5% to 5%)
- Evaluation mode, which reveals the true labels (off by default, as an analyst would not have them)

The **Alert details** tab shows why a chosen alert was raised: the features that pushed the XGBoost score up (TreeSHAP values) and the features the autoencoder reconstructed worst, each beside the normal Monday value.

The dashboard scores each test day live from the saved models and checks the result against the notebook's saved scores. The first load takes about a minute.

## Environment used for the reported results

Python 3.13.13, pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.9.1, XGBoost 3.4.1, PyTorch 2.15.0 (CPU), matplotlib 3.11.2, on macOS. Random seed 42 throughout. Notebook 04 recorded NumPy 2.5.3. Its random-split XGBoost model differs very slightly from CW1's (MCC 0.99719 against 0.99721, see `outputs/04_reproducibility_vs_cw1.csv`), most likely because of library version differences between the two runs.

## Limitations

This is an academic prototype, not a production system. It uses a public lab dataset (CICIDS2017) in place of TeleComNet traffic, scores individual flows rather than hosts or sessions, and was run with a single random seed. These points are discussed in the technical documentation.
