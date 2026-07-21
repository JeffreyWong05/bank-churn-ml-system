"""
Production monitoring simulation.

A model that ships is a model you have to watch. This module stands in for the
monitoring job that would run on a schedule against fresh scoring data:

  1. Population Stability Index (PSI) per feature, comparing a live batch to
     the training reference distribution. PSI > 0.1 = moderate drift,
     > 0.25 = significant drift (the usual rule-of-thumb thresholds).
  2. Rolling performance: ROC-AUC computed on sequential batches so you can
     see decay before it hurts the business.

To make the demo concrete, simulate_stream() carves the test set into time
batches and injects gradual covariate drift into the later ones (ageing +
rising balances), which is exactly the kind of slow shift that erodes a churn
model in the field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    ref_pct = np.histogram(ref, bins=quantiles)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=quantiles)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame, features):
    out = {}
    for f in features:
        if str(reference[f].dtype) == "category":
            r = reference[f].cat.codes.values
            c = current[f].cat.codes.values
        else:
            r, c = reference[f].values, current[f].values
        out[f] = round(psi(r, c), 4)
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def _label(v):
    return "significant" if v > 0.25 else ("moderate" if v > 0.1 else "stable")


def simulate_stream(model, X_ref, X_test, y_test, n_batches: int = 6, seed: int = 42):
    """Split test data into batches, inject growing drift, track AUC + PSI."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_test))
    batches = np.array_split(idx, n_batches)
    numeric = ["Age", "Balance", "CreditScore", "EstimatedSalary",
               "BalanceSalaryRatio"]

    rows = []
    for i, b in enumerate(batches):
        Xb = X_test.iloc[b].copy()
        yb = y_test.iloc[b]
        # Inject gradual drift into later batches: customers age and balances rise.
        drift = i / (n_batches - 1)
        Xb["Age"] = Xb["Age"] + rng.normal(6 * drift, 1, len(Xb))
        Xb["Balance"] = Xb["Balance"] * (1 + 0.4 * drift)
        Xb["BalanceSalaryRatio"] = Xb["Balance"] / (Xb["EstimatedSalary"] + 1.0)

        proba = model.predict_proba(Xb)[:, 1]
        auc = roc_auc_score(yb, proba) if yb.nunique() > 1 else float("nan")
        psis = feature_psi(X_ref, Xb, numeric)
        worst_feat, worst_val = max(psis.items(), key=lambda kv: kv[1])
        rows.append({
            "batch": i + 1,
            "n": int(len(b)),
            "auc": round(float(auc), 4),
            "max_psi": worst_val,
            "max_psi_feature": worst_feat,
            "drift_status": _label(worst_val),
            "psi_by_feature": psis,
        })
    return pd.DataFrame(rows)
