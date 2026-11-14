"""
7CS537 CW2 artefact: TeleComNet hybrid alert triage dashboard (academic prototype).
Student number: 100810453

What it does
- Loads the CICIDS2017 flows of one test day and scores every flow LIVE with the two trained models:
  XGBoost (Notebook 04, trained only on earlier days) and the autoencoder (Notebook 05, trained on benign
  Monday traffic), run here in NumPy.
- Sets the autoencoder alert threshold from benign traffic of earlier days only, for a chosen false alarm target.
- Ranks the day's flows into a queue capped at the analyst team's daily capacity, using one of three strategies.
- Explains any alert: XGBoost SHAP values (TreeSHAP) and the autoencoder's worst-rebuilt features, each shown
  next to the normal (benign Monday median) value.
- Evaluation mode reveals the true labels, so the queue's real attacks can be counted. An analyst would not
  have these labels; they are hidden by default.

Run from the repository folder, after Notebooks 03 to 06:  streamlit run cw2/app/triage_dashboard.py
PyTorch is deliberately not imported: it crashes alongside XGBoost on macOS. The autoencoder weights were
converted to NumPy by Notebook 06 (cw2/models/05_autoencoder_weights.npz).
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
import altair as alt
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CW2 = os.path.join(ROOT, "cw2")
MODELS = os.path.join(CW2, "models")
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TEST_DAYS = {"Friday": "E1", "Wednesday": "E2_Wednesday", "Thursday": "E2_Thursday"}
STRATEGIES = ["Hybrid tiers", "XGBoost score", "Autoencoder error"]

st.set_page_config(page_title="TeleComNet alert triage", layout="wide")


# ---------- Loading (cached so it happens once) ----------
@st.cache_data(show_spinner="Loading CICIDS2017 flows and splits...")
def load_data():
    df = pd.read_parquet(os.path.join(ROOT, "data", "cicids2017_clean.parquet"))
    with open(os.path.join(ROOT, "outputs", "01_feature_list.txt")) as fh:
        features = [line.strip() for line in fh if line.strip()]
    splits = pd.read_parquet(os.path.join(CW2, "data", "03_splits.parquet"))
    X = df[features].to_numpy(dtype=np.float32)
    meta = pd.DataFrame({"day": splits["day"].astype(str).to_numpy(),
                         "ae_role": splits["ae_role"].astype(str).to_numpy(),
                         "Label": df["Label"].astype(str).to_numpy(),
                         "is_attack": df["is_attack"].to_numpy().astype(int)})
    saved_xgb = pd.read_parquet(os.path.join(CW2, "data", "04_xgb_test_scores.parquet"))
    saved_ae = pd.read_parquet(os.path.join(CW2, "data", "05_anomaly_scores.parquet"))["ae_error"].to_numpy()
    normal_median = np.median(X[meta["ae_role"].to_numpy() == "ae_train"], axis=0)
    return X, features, meta, saved_xgb, saved_ae, normal_median


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    xgb_models = {}
    for d, exp in TEST_DAYS.items():
        m = XGBClassifier()
        m.load_model(os.path.join(MODELS, f"04_xgb_{exp}.json"))
        xgb_models[d] = m
    w = np.load(os.path.join(MODELS, "05_autoencoder_weights.npz"))
    names = sorted({k.rsplit(".", 1)[0] for k in w.files},
                   key=lambda n: (n.split(".")[0] != "encoder", int(n.split(".")[1])))
    layers = [(w[f"{n}.weight"], w[f"{n}.bias"]) for n in names]
    prep = joblib.load(os.path.join(MODELS, "05_preprocess.joblib"))
    return xgb_models, layers, prep


X, FEATURES, META, SAVED_XGB, SAVED_AE, NORMAL_MEDIAN = load_data()
XGB_MODELS, LAYERS, PREP = load_models()
FEAT = np.array(FEATURES)


def preprocess(x):
    z = np.sign(x) * np.log1p(np.abs(x))
    return np.clip(PREP["scaler"].transform(z), -PREP["clip"], PREP["clip"]).astype(np.float32)


def ae_feature_errors(rows):
    xs = preprocess(X[rows])
    h = xs
    for i, (w, b) in enumerate(LAYERS):
        h = h @ w.T + b
        if i < len(LAYERS) - 1:
            h = np.maximum(h, 0)
    return (h - xs) ** 2


@st.cache_data(show_spinner="Scoring every flow of the day with both models...")
def score_day(day):
    idx = np.where(META["day"].to_numpy() == day)[0]
    xgb_p = XGB_MODELS[day].predict_proba(X[idx])[:, 1]
    ae = np.concatenate([ae_feature_errors(idx[i:i + 200000]).mean(axis=1) for i in range(0, len(idx), 200000)])
    check = {"xgb": float(np.max(np.abs(xgb_p - SAVED_XGB[TEST_DAYS[day]].to_numpy()[idx]))),
             "ae": float(np.max(np.abs(ae - SAVED_AE[idx]) / np.maximum(SAVED_AE[idx], 1e-6)))}
    return idx, xgb_p, ae, check


@st.cache_data
def ae_threshold(day, target):
    # Benign flows before the test day only: Monday hold-out + benign Tuesday up to the day before
    before = DAY_ORDER[1:DAY_ORDER.index(day)]
    m = (META["ae_role"] == "ae_val") | (META["day"].isin(before) & (META["is_attack"] == 0))
    cal = np.where(m.to_numpy())[0]
    errs = np.concatenate([ae_feature_errors(cal[i:i + 200000]).mean(axis=1) for i in range(0, len(cal), 200000)])
    return float(np.quantile(errs, 1 - target)), len(cal)


def rank(strategy, xgb_p, ae, thr):
    xf, af = xgb_p >= 0.5, ae >= thr
    tier = np.where(xf & af, 1, np.where(xf, 2, np.where(af, 3, 4)))
    if strategy == "XGBoost score":
        order = np.argsort(-xgb_p, kind="stable")
    elif strategy == "Autoencoder error":
        order = np.argsort(-ae, kind="stable")
    else:
        order = np.lexsort((np.where(tier <= 2, -xgb_p, -ae), tier))
    return order, tier


# ---------- Sidebar ----------
st.sidebar.title("Analyst settings")
day = st.sidebar.selectbox("Test day", list(TEST_DAYS), index=0,
                           help="XGBoost for each day was trained only on earlier days.")
strategy = st.sidebar.radio("Rank alerts by", STRATEGIES, index=0,
                            help="Hybrid tiers: P1 both models, P2 XGBoost only, P3 autoencoder only.")
capacity = st.sidebar.slider("Alerts the team can review today", 100, 5000, 1000, step=100)
target = st.sidebar.select_slider("Autoencoder false alarm target", options=[0.005, 0.01, 0.02, 0.05], value=0.01,
                                  format_func=lambda v: f"{v:.1%}",
                                  help="Threshold = this percentile of benign traffic from earlier days.")
evaluation = st.sidebar.toggle("Evaluation mode (show true labels)", value=False,
                               help="Labels are hidden by default because an analyst would not have them.")
st.sidebar.caption("Academic prototype for 7CS537 CW2. CICIDS2017 stands in for TeleComNet traffic. "
                   "Not production-ready.")

# ---------- Scoring and queue ----------
idx, xgb_p, ae, check = score_day(day)
thr, n_cal = ae_threshold(day, target)
order, tier = rank(strategy, xgb_p, ae, thr)
top = order[:capacity]
rows = idx[top]
y = META["is_attack"].to_numpy()

st.title("TeleComNet SOC: hybrid alert triage")
st.caption(f"{day}: {len(idx):,} flows scored live with XGBoost and the autoencoder. "
           f"Autoencoder threshold {thr:.3f} from {n_cal:,} benign flows of earlier days ({target:.1%} target).")

c = st.columns(5)
c[0].metric("Flows today", f"{len(idx):,}")
c[1].metric("Flagged by XGBoost", f"{int((xgb_p >= 0.5).sum()):,}", help="Flows with an XGBoost score of 0.5 or more.")
c[2].metric("Flagged by autoencoder", f"{int((ae >= thr).sum()):,}")
c[3].metric("In today's queue", f"{len(rows):,}")
if evaluation:
    real = int(y[rows].sum())
    c[4].metric("Real attacks in queue", f"{real:,}", f"{real / len(rows):.1%} precision", delta_color="off")
else:
    c[4].metric("Real attacks in queue", "hidden", help="Turn on evaluation mode to see labels.")

ok = check["xgb"] < 1e-4 and check["ae"] < 1e-3
st.caption(("✅ " if ok else "⚠️ ") + f"Live scores match the notebook results (largest difference: XGBoost "
           f"{check['xgb']:.1e}, autoencoder {check['ae']:.1e} relative).")

tab_queue, tab_detail, tab_about = st.tabs(["Alert queue", "Alert details", "About this prototype"])

with tab_queue:
    q = pd.DataFrame({"Rank": np.arange(1, len(rows) + 1), "Tier": [f"P{t}" for t in tier[top]],
                      "XGBoost score": np.round(xgb_p[top], 4), "Autoencoder error": np.round(ae[top], 4)})
    pfe = ae_feature_errors(rows)
    q["Most unusual feature (autoencoder)"] = FEAT[np.argmax(pfe, axis=1)]
    if evaluation:
        q["True label"] = META["Label"].to_numpy()[rows]
    st.dataframe(q, hide_index=True, width="stretch", height=420)
    if evaluation:
        counts = q.loc[q["True label"] != "BENIGN", "True label"].value_counts()
        st.write("**Attack types in the queue:** " + (", ".join(f"{k} {v:,}" for k, v in counts.items()) or "none"))
    tiers = pd.Series([f"P{t}" for t in tier]).value_counts().reindex(["P1", "P2", "P3"], fill_value=0)
    st.caption(f"Whole day, before capacity: P1 {tiers['P1']:,}, P2 {tiers['P2']:,}, P3 {tiers['P3']:,} alerts.")

with tab_detail:
    r = st.number_input("Queue rank to inspect", min_value=1, max_value=len(rows), value=1, step=1)
    k = top[r - 1]
    flow = idx[k]
    a, b, c3 = st.columns(3)
    a.metric("Tier", f"P{tier[k]}")
    b.metric("XGBoost score", f"{xgb_p[k]:.4f}", "above 0.5" if xgb_p[k] >= 0.5 else "below 0.5", delta_color="off")
    c3.metric("Autoencoder error", f"{ae[k]:.4f}", f"threshold {thr:.3f}", delta_color="off")
    if evaluation:
        st.info(f"True label: **{META['Label'].to_numpy()[flow]}**")

    shap = XGB_MODELS[day].get_booster().predict(xgb.DMatrix(X[[flow]]), pred_contribs=True)[0, :-1]
    err = ae_feature_errors(np.array([flow]))[0]
    left, right = st.columns(2)
    for col, title, values, note in [
        (left, "Why XGBoost scored it (SHAP, log-odds)", shap, "Positive pushes towards attack."),
        (right, "What the autoencoder found unusual", err, "Squared error in standard deviations of normal traffic.")]:
        j = np.argsort(-np.abs(values) if values is shap else -values)[:8]
        d = pd.DataFrame({"Feature": FEAT[j], "Contribution": values[j],
                          "This flow": [f"{X[flow, i]:,.6g}" for i in j],
                          "Normal median": [f"{NORMAL_MEDIAN[i]:,.6g}" for i in j]})
        chart = alt.Chart(d).mark_bar().encode(
            x=alt.X("Contribution:Q"), y=alt.Y("Feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.condition("datum.Contribution > 0", alt.value("#C44E52"), alt.value("#4C72B0")),
            tooltip=["Feature", "Contribution", "This flow", "Normal median"])
        col.subheader(title)
        col.altair_chart(chart, width="stretch")
        col.dataframe(d, hide_index=True, width="stretch")
        col.caption(note)
    st.caption("Explanations show what drove each model's score. They are a starting point for investigation, "
               "not proof of cause; related features can share credit.")

with tab_about:
    st.markdown("""
**Problem.** TeleComNet's SOC needs to notice new attack campaigns without drowning analysts in alerts.

**Pipeline.** CICIDS2017 flows (62 features, cleaned in CW1) → time-ordered split by capture day → XGBoost
trained on earlier days (known attacks) + autoencoder trained on benign Monday traffic (unusual traffic) →
benign-only threshold → priority tiers → daily queue capped at analyst capacity → explanation per alert.

**Key results (Notebooks 04 to 06).** XGBoost trained on earlier days caught 0.12% of Friday's new attacks at
its 0.5 threshold. The autoencoder caught 78.4% of Wednesday's new DoS attacks at a 1.2% false positive rate,
but Friday's normal traffic drifted (6.7% false positive rate). Ranking against capacity worked better than fixed
thresholds, and no single ranking was best on every day.

**Limitations.** 2017 lab data; flow-by-flow view misses scans and brute force that are only visible across many
flows; thresholds drift between days; no adversarial testing; not production-ready.
""")
