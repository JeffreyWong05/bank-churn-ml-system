# Make It Yours — Extension Ideas

Do at least one of these so the project is demonstrably your work (and your
commit history shows it). Each is scoped to a few hours and maps to something
the job description actually asks for.

---

## 1. Model bake-off: add XGBoost and compare
**Why it impresses:** shows you don't just reach for one algorithm — you compare
and justify. Directly supports "rigorous model evaluation."

Create `src/compare_models.py`:
```python
from data import prepare
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier   # add xgboost to requirements.txt
import os

d = prepare(os.path.join(os.path.dirname(__file__), "..", "data", "churn.csv"))

# XGBoost needs numeric input; encode the category columns as codes.
def encode(X):
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes
    return X

results = {}
for name, model in {
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
}.items():
    Xtr, Xte = encode(d["X_train"]), encode(d["X_test"])
    model.fit(Xtr, d["y_train"])
    auc = roc_auc_score(d["y_test"], model.predict_proba(Xte)[:, 1])
    results[name] = round(auc, 4)

print("ROC-AUC by model:", results)
```
Then add a short "Model comparison" note to the README with the numbers.

---

## 2. Decision-threshold tuner
**Why it impresses:** 0.5 is rarely the right cutoff. Choosing a threshold to
maximise *business value* (not accuracy) is exactly the applied-ML judgment the
role wants.

Add to `src/business.py`:
```python
import numpy as np

def best_threshold(proba, y_true, annual_value=1200.0, offer_cost=40.0,
                   save_rate=0.30):
    """Sweep thresholds; return the one that maximises net campaign value."""
    proba, y_true = np.asarray(proba), np.asarray(y_true)
    best = {"threshold": 0.5, "net_value": -1e9}
    for t in np.linspace(0.05, 0.9, 50):
        flagged = proba >= t
        caught = int((flagged & (y_true == 1)).sum())
        cost = int(flagged.sum()) * offer_cost
        net = caught * save_rate * annual_value - cost
        if net > best["net_value"]:
            best = {"threshold": round(float(t), 3), "net_value": round(net, 0)}
    return best
```
Then surface it in the dashboard's Business tab with a slider:
```python
t = st.slider("Outreach threshold", 0.05, 0.9, 0.5, 0.05)
st.metric("Customers flagged", int((proba_test >= t).sum()))
```

---

## 3. Add a feature and measure its SHAP impact
**Why it impresses:** shows the full loop — hypothesise a feature, add it,
re-measure importance.

In `src/data.py`, inside `engineer()`, add one line:
```python
# "Sleeping money": high balance but inactive — a plausible churn signal.
df["IdleHighBalance"] = ((df["Balance"] > df["Balance"].median()) &
                         (df["IsActiveMember"] == 0)).astype(int)
```
Re-run `python src/run_pipeline.py`, then open `reports/shap_global.json` and
note where your new feature ranks. Write one sentence in the README about
whether it helped.

---

## 4. Replace the assumed save-rate with a simulated A/B holdout
**Why it impresses:** turns an assumption into a *measured* number with
uncertainty — the honest version of "we'd learn this from an experiment."

Create `src/ab_sim.py`:
```python
import numpy as np

def simulate_ab(proba, y_true, depth=0.10, true_uplift=0.08, seed=42):
    """Split the targeted top-`depth` into treatment/control and estimate the
    retention uplift the way an A/B test would, instead of assuming it."""
    rng = np.random.default_rng(seed)
    proba, y_true = np.asarray(proba), np.asarray(y_true)
    k = int(depth * len(proba))
    targeted = np.argsort(-proba)[:k]
    treat = rng.random(k) < 0.5
    churned = y_true[targeted] == 1
    # Treatment reduces churn among would-be churners by `true_uplift`.
    saved_treat = (churned & treat & (rng.random(k) < true_uplift)).sum()
    control_churn = (churned & ~treat).mean()
    treat_churn = (churned & treat).mean()
    return {
        "measured_uplift": round(float(control_churn - treat_churn), 4),
        "customers_saved_est": int(saved_treat),
    }
```
Mention in the README that the business case can be driven by a *measured*
uplift, not just an assumed save-rate.

---

### After any extension
```bash
git add -A
git commit -m "Add <thing you built>"
git push
```
Streamlit Community Cloud auto-redeploys within a minute of the push.
