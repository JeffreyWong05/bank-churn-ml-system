"""
Run the full pipeline end to end and write every artifact the dashboard and
README consume. This is the single command a reviewer runs to reproduce the
project from raw data:

    python src/run_pipeline.py

Produces under reports/ and models/:
    model.joblib, metrics.json, shap_summary.png, shap_global.json,
    fairness_report.json, fairness_report.md, monitoring.json,
    business_curve.csv, summary.json
"""
from __future__ import annotations

import json
import os
import pandas as pd

import data as data_mod
import model as model_mod
import fairness as fair_mod
import interpret as shap_mod
import monitor as mon_mod
import business as biz_mod

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "churn.csv")
REPORTS = os.path.join(HERE, "reports")
MODELS = os.path.join(HERE, "models")


def main(n_trials: int = 25):
    os.makedirs(REPORTS, exist_ok=True)
    print("1/6  Loading + engineering features ...")
    d = data_mod.prepare(DATA)

    print("2/6  Training LightGBM (Optuna) + calibrating ...")
    calibrated, base, metrics = model_mod.train(d, n_trials=n_trials, out_dir=MODELS)
    print(f"     ROC-AUC={metrics['roc_auc']}  PR-AUC={metrics['pr_auc']}")

    proba = calibrated.predict_proba(d["X_test"])[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("3/6  Fairness audit (Gender, Geography) ...")
    fair = fair_mod.audit(d["y_test"], preds, d["prot_test"], out_dir=REPORTS)
    with open(os.path.join(REPORTS, "fairness_report.md"), "w") as f:
        f.write(fair_mod.to_markdown(fair))

    print("4/6  SHAP interpretability ...")
    ranking = shap_mod.explain_global(base, d["X_test"], out_dir=REPORTS)
    top_features = list(ranking.head(5).index)

    print("5/6  Monitoring / drift simulation ...")
    stream = mon_mod.simulate_stream(calibrated, d["X_train"], d["X_test"],
                                     d["y_test"])
    stream_out = stream.drop(columns=["psi_by_feature"]).to_dict(orient="records")
    with open(os.path.join(REPORTS, "monitoring.json"), "w") as f:
        json.dump({"batches": stream_out,
                   "psi_detail": stream["psi_by_feature"].tolist()}, f, indent=2)

    print("6/6  Business value curve + uplift vs random ...")
    curve = biz_mod.expected_value_curve(proba, d["y_test"])
    curve.to_csv(os.path.join(REPORTS, "business_curve.csv"), index=False)
    uplift = biz_mod.uplift_vs_random(proba, d["y_test"], depth=0.10)

    summary = {
        "headline_metrics": {k: metrics[k] for k in
                             ["roc_auc", "pr_auc", "recall", "precision", "brier"]},
        "top_drivers": top_features,
        "fairness": {a: {
            "demographic_parity_difference": fair[a]["demographic_parity_difference"],
            "equal_opportunity_difference": fair[a]["equal_opportunity_difference"],
        } for a in fair},
        "monitoring_final_batch": stream_out[-1],
        "best_targeting": biz_mod.expected_value_curve(proba, d["y_test"])
            .sort_values("net_value", ascending=False).iloc[0].to_dict(),
        "uplift_vs_random_at_10pct": uplift,
    }
    with open(os.path.join(REPORTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\nDONE. Key results:")
    print(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    main()
