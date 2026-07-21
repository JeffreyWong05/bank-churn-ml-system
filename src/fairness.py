"""
Fairness audit for the churn model.

A retention model that flags customers for outreach is a lower-stakes decision
than credit, but in a regulated bank *any* model that scores customers should
be checked for disparate behaviour across protected groups. We report, per
group of a protected attribute:

  * selection_rate  - share flagged as likely to churn (demographic-parity view)
  * recall (TPR)    - of the customers who actually churned, how many we caught
  * fpr             - false-positive rate (wasted outreach on loyal customers)
  * precision       - of those we flagged, how many actually churned

and then summarise the gaps as:

  * demographic_parity_difference = max(selection_rate) - min(selection_rate)
  * equal_opportunity_difference  = max(recall) - min(recall)

These are the standard group-fairness definitions (Hardt et al., 2016). We do
not "fix" anything automatically; the point of the audit is to surface the
disparity so a human decides whether it is acceptable and documented.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd


def _group_metrics(y_true, y_pred, mask):
    yt, yp = y_true[mask], y_pred[mask]
    n = int(mask.sum())
    pos = int((yt == 1).sum())
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    flagged = int((yp == 1).sum())
    neg = n - pos
    return {
        "n": n,
        "base_churn_rate": round(pos / n, 4) if n else 0.0,
        "selection_rate": round(flagged / n, 4) if n else 0.0,
        "recall": round(tp / pos, 4) if pos else 0.0,
        "fpr": round(fp / neg, 4) if neg else 0.0,
        "precision": round(tp / flagged, 4) if flagged else 0.0,
    }


def audit(y_true, y_pred, protected: pd.DataFrame, out_dir: str = "reports"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = {}

    for attr in protected.columns:
        groups = {}
        col = protected[attr].values
        for g in pd.unique(col):
            groups[str(g)] = _group_metrics(y_true, y_pred, col == g)

        sel = [v["selection_rate"] for v in groups.values()]
        rec = [v["recall"] for v in groups.values()]
        report[attr] = {
            "groups": groups,
            "demographic_parity_difference": round(max(sel) - min(sel), 4),
            "equal_opportunity_difference": round(max(rec) - min(rec), 4),
        }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fairness_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def to_markdown(report: dict) -> str:
    lines = ["# Fairness Audit\n"]
    for attr, block in report.items():
        lines.append(f"## By {attr}\n")
        lines.append("| Group | n | Base churn | Selection rate | Recall (TPR) | FPR | Precision |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for g, m in block["groups"].items():
            lines.append(
                f"| {g} | {m['n']} | {m['base_churn_rate']:.3f} | "
                f"{m['selection_rate']:.3f} | {m['recall']:.3f} | "
                f"{m['fpr']:.3f} | {m['precision']:.3f} |"
            )
        lines.append("")
        lines.append(f"- **Demographic-parity difference:** {block['demographic_parity_difference']:.3f}")
        lines.append(f"- **Equal-opportunity difference:** {block['equal_opportunity_difference']:.3f}\n")
    return "\n".join(lines)
