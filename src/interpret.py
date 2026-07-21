"""
Model interpretability with SHAP.

We explain the *uncalibrated* LightGBM booster (the calibration wrapper is a
monotone transform of its scores, so feature attributions are identical in
rank and nearly identical in magnitude, and TreeExplainer is exact and fast
on the raw booster).

Outputs:
  * reports/shap_summary.png     - global feature importance (mean |SHAP|)
  * reports/shap_global.json     - the same ranking as data for the dashboard
  * a local_explanation() helper - per-customer top drivers for the app
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _coded(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric-coded copy used only for the summary plot's colour axis."""
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes
    return X


def explain_global(base_model, X_sample: pd.DataFrame, out_dir: str = "reports",
                   max_display: int = 12):
    # SHAP's TreeExplainer must see the same category dtypes the booster trained
    # on, so we pass X_sample unchanged and only code it for plot colouring.
    explainer = shap.TreeExplainer(base_model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):          # older shap returns [class0, class1]
        sv = sv[1]

    mean_abs = np.abs(sv).mean(axis=0)
    ranking = (pd.Series(mean_abs, index=X_sample.columns)
               .sort_values(ascending=False))

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, _coded(X_sample), feature_names=list(X_sample.columns),
                      show=False, max_display=max_display, plot_size=(8, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=130)
    plt.close()

    with open(os.path.join(out_dir, "shap_global.json"), "w") as f:
        json.dump({k: round(float(v), 5) for k, v in ranking.items()}, f, indent=2)
    return ranking


def local_explanation(base_model, x_row: pd.DataFrame, top_k: int = 5):
    """Return the top_k signed SHAP drivers for a single customer row."""
    explainer = shap.TreeExplainer(base_model)
    sv = explainer.shap_values(x_row)
    if isinstance(sv, list):
        sv = sv[1]
    contrib = pd.Series(np.asarray(sv)[0], index=x_row.columns)
    ordered = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    return ordered.head(top_k)
