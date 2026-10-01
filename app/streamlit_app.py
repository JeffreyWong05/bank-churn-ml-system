"""
Bank Customer Churn — Applied ML System dashboard.

Run locally:   streamlit run app/streamlit_app.py
Deploy free:   push to GitHub, then Streamlit Community Cloud -> this file.

The dashboard is the "show, don't tell" surface for the whole project:
score a customer live, see why the model flagged them (SHAP), audit fairness
across protected groups, watch the model drift under simulated production
data, and read the retention-campaign business case.
"""
import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import recall_score, precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data as data_mod          # noqa: E402
import interpret as shap_mod     # noqa: E402
import business as biz_mod       # noqa: E402

st.set_page_config(page_title="Bank Churn — Applied ML System",
                   layout="wide", page_icon="🏦")


@st.cache_resource
def load_all():
    bundle = joblib.load(os.path.join(ROOT, "models", "model.joblib"))
    d = data_mod.prepare(os.path.join(ROOT, "data", "churn.csv"))
    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(ROOT, "reports", "fairness_report.json")) as f:
        fairness = json.load(f)
    with open(os.path.join(ROOT, "reports", "monitoring.json")) as f:
        monitoring = json.load(f)
    return bundle, d, metrics, fairness, monitoring


bundle, D, METRICS, FAIRNESS, MONITORING = load_all()
model, base = bundle["model"], bundle["base"]
proba_test = model.predict_proba(D["X_test"])[:, 1]

st.title("🏦 Bank Customer Churn — Applied ML System")
st.caption("LightGBM churn model with SHAP interpretability, a fairness audit, "
           "production drift monitoring, and a retention-campaign business case. "
           "Built on 10,000 bank customers.")

# ---- Decision threshold (shared across tabs) --------------------------------
# Change this one value to set the threshold the app opens with.
DEFAULT_THRESHOLD = 0.50

def _sync_slider_to_num():
    st.session_state.thr_num = st.session_state.thr_slider

def _sync_num_to_slider():
    st.session_state.thr_slider = st.session_state.thr_num

if "thr_slider" not in st.session_state:
    st.session_state.thr_slider = DEFAULT_THRESHOLD
    st.session_state.thr_num = DEFAULT_THRESHOLD

st.sidebar.header("Decision threshold")
st.sidebar.caption("Flag a customer as 'will churn' when their probability is at "
                   "or above this cutoff. Lower it to catch more churners "
                   "(higher recall) at the cost of more false alarms.")
st.sidebar.slider("Threshold (drag)", 0.05, 0.95, step=0.01,
                  key="thr_slider", on_change=_sync_slider_to_num)
st.sidebar.number_input("…or type an exact value", 0.05, 0.95, step=0.01,
                        key="thr_num", on_change=_sync_num_to_slider)
THRESHOLD = st.session_state.thr_slider
# -----------------------------------------------------------------------------

tab_pred, tab_shap, tab_fair, tab_mon, tab_biz = st.tabs(
    ["🎯 Score a customer", "🔍 Why (SHAP)", "⚖️ Fairness",
     "📈 Monitoring", "💰 Business case"])

# ------------------------------------------------------------------ Predict
with tab_pred:
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", METRICS["roc_auc"])
    c2.metric("PR-AUC", METRICS["pr_auc"])
    c3.metric("Brier (calibration)", METRICS["brier"])

    st.subheader("Enter a customer profile")
    a, b, c = st.columns(3)
    credit = a.slider("Credit score", 350, 850, 650)
    age = b.slider("Age", 18, 92, 40)
    tenure = c.slider("Tenure (yrs)", 0, 10, 5)
    balance = a.number_input("Balance", 0.0, 260000.0, 75000.0, step=1000.0)
    salary = b.number_input("Estimated salary", 0.0, 200000.0, 100000.0, step=1000.0)
    products = c.selectbox("Num products", [1, 2, 3, 4], index=0)
    geo = a.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = b.selectbox("Gender", ["Female", "Male"])
    active = c.selectbox("Active member", [1, 0])
    has_card = a.selectbox("Has credit card", [1, 0])

    row = pd.DataFrame([{
        "CreditScore": credit, "Geography": geo, "Gender": gender, "Age": age,
        "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
        "HasCrCard": has_card, "IsActiveMember": active, "EstimatedSalary": salary,
    }])
    row = data_mod.engineer(row)
    row = row[D["feature_names"]]
    for col in D["categorical"]:
        row[col] = pd.Categorical(row[col],
                                  categories=D["X_train"][col].cat.categories)

    p = float(model.predict_proba(row)[:, 1][0])
    st.markdown("### Predicted churn probability")
    st.progress(min(p, 1.0))
    verdict = "⚠️ High risk — recommend retention outreach" if p >= THRESHOLD \
        else "✅ Low risk — no action needed"
    st.metric("Churn probability", f"{p:.1%}", verdict)
    st.caption(f"Classified using the decision threshold {THRESHOLD:.2f} "
               "(set it in the sidebar).")

    st.markdown("#### Top drivers for *this* customer")
    local = shap_mod.local_explanation(base, row, top_k=5)
    drivers = pd.DataFrame({
        "feature": local.index,
        "pushes_toward": ["churn ↑" if v > 0 else "staying ↓" for v in local.values],
        "shap_value": local.values.round(3),
    })
    st.dataframe(drivers, hide_index=True, use_container_width=True)

    st.markdown(f"#### Model performance at threshold {THRESHOLD:.2f} "
                "(whole test set)")
    preds_thr = (proba_test >= THRESHOLD).astype(int)
    yt = D["y_test"].values
    m1, m2, m3 = st.columns(3)
    m1.metric("Recall (churners caught)", f"{recall_score(yt, preds_thr):.1%}")
    m2.metric("Precision (flags correct)",
              f"{precision_score(yt, preds_thr, zero_division=0):.1%}")
    m3.metric("Share of customers flagged", f"{preds_thr.mean():.1%}")
    st.caption("Drag the sidebar threshold and watch the trade-off: recall rises "
               "as you lower the cutoff, while precision falls. This is why the "
               "0.50 default is a business lever, not a fixed rule.")

# ------------------------------------------------------------------ SHAP
with tab_shap:
    st.subheader("Global feature importance (mean |SHAP|)")
    st.image(os.path.join(ROOT, "reports", "shap_summary.png"),
             use_container_width=True)
    st.markdown(
            "Each dot is a customer. Position shows how strongly a feature pushed "
            "that customer's churn score up (right) or down (left); colour is the "
            "feature value. This is how the model's logic is made auditable rather "
            "than a black box.")
        # --- Feature interaction view ---
    st.subheader("Feature interactions")          
    xt = D["X_test"].copy()
    xt["churned"] = D["y_test"].values
    med = xt.loc[xt["BalanceSalaryRatio"] > 0, "BalanceSalaryRatio"].median()
    xt["BSR_band"] = xt["BalanceSalaryRatio"].map(
        lambda v: "empty" if v == 0 else ("low" if v < med else "high"))
    piv = (xt.pivot_table(index="NumOfProducts", columns="BSR_band",
                          values="churned", aggfunc="mean") * 100).round(1)
    st.dataframe(piv.reindex(columns=["empty", "low", "high"]),
                 use_container_width=True)
    st.caption("The Balance to Salary ratio reverses in direction as product count increases. An empty balance is risky at 1 product," \
    "but safe when at two products. The conditional situation is the interaction.")
    

# ------------------------------------------------------------------ Fairness
with tab_fair:
    st.subheader("Group-fairness audit across protected attributes")
    st.caption("Definitions: demographic-parity difference = gap in flag rates; "
               "equal-opportunity difference = gap in recall (churners actually "
               "caught). Thresholds are judgement calls a human documents.")
    for attr, block in FAIRNESS.items():
        st.markdown(f"#### By {attr}")
        rows = []
        for g, m in block["groups"].items():
            rows.append({"group": g, **m})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        dp = block["demographic_parity_difference"]
        eo = block["equal_opportunity_difference"]
        flag = "🔴" if eo > 0.1 else "🟢"
        st.markdown(f"{flag} **Demographic-parity difference:** {dp:.3f}  |  "
                    f"**Equal-opportunity difference:** {eo:.3f}")
    st.info("Finding: recall parity is strong across gender but weaker across "
            "geography — the kind of disparity a regulated lender must surface, "
            "document, and decide on before deployment.")

# ------------------------------------------------------------------ Monitoring
with tab_mon:
    st.subheader("Production drift monitoring (simulated stream)")
    batches = pd.DataFrame(MONITORING["batches"])
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Rolling ROC-AUC by batch**")
        st.line_chart(batches.set_index("batch")["auc"])
    with m2:
        st.markdown("**Max feature PSI by batch** (>0.25 = significant drift)")
        st.line_chart(batches.set_index("batch")["max_psi"])
    st.dataframe(
        batches[["batch", "n", "auc", "max_psi", "max_psi_feature",
                 "drift_status"]],
        hide_index=True, use_container_width=True)
    st.warning("Injected covariate drift (ageing customers, rising balances) "
               "erodes AUC and trips the PSI alarm on Balance — exactly the "
               "signal a monitoring job should page on.")

# ------------------------------------------------------------------ Business
with tab_biz:
    st.subheader("Retention-campaign business case")
    c1, c2, c3 = st.columns(3)
    annual = c1.number_input("Annual value / retained customer ($)", 200, 5000, 1200, 100)
    cost = c2.number_input("Outreach cost / customer ($)", 5, 200, 40, 5)
    save = c3.slider("Assumed save rate (campaign uplift)", 0.05, 0.6, 0.30, 0.05)

    curve = biz_mod.expected_value_curve(proba_test, D["y_test"].values,
                                         annual_value=annual, offer_cost=cost,
                                         save_rate=save)
    st.markdown("**Expected value by targeting depth** "
                "(rank customers by churn risk, call the top X%)")
    st.dataframe(curve, hide_index=True, use_container_width=True)
    st.bar_chart(curve.set_index("target_depth")["net_value"])

    up = biz_mod.uplift_vs_random(proba_test, D["y_test"].values, depth=0.10,
                                  annual_value=annual, offer_cost=cost,
                                  save_rate=save)
    st.metric("Net value uplift vs a random call list (top 10%)",
              f"${up['uplift']:,.0f}",
              f"model ${up['model_net_value']:,.0f} vs random ${up['random_net_value']:,.0f}")
    st.caption("In production the save rate would be learned from an A/B "
               "holdout rather than assumed — the framework is built for it.")
