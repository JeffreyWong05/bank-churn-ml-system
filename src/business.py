"""
Translate churn probabilities into a business decision.

A churn score is not a deliverable; a *retention plan* is. This module answers
the question a product owner actually asks: "if I can only afford to call the
top K% of customers, who do I call, and what's the expected return?"

Assumptions are explicit and adjustable (they'd be owned by Finance in real
life):
  * annual_value   - contribution margin retained if a churner stays
  * offer_cost     - cost of the retention offer/outreach per targeted customer
  * save_rate      - fraction of *correctly targeted* churners actually retained
                     (this is the campaign uplift; in production you'd learn it
                     from an A/B holdout instead of assuming it)

expected_value_curve() ranks customers by predicted churn probability and, for
each targeting depth, reports precision, expected saves, cost, and net value —
a decision curve a stakeholder can read without knowing what ROC-AUC means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_value_curve(proba, y_true, annual_value=1200.0,
                         offer_cost=40.0, save_rate=0.30,
                         depths=None):
    proba = np.asarray(proba)
    y_true = np.asarray(y_true)
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    n = len(proba)
    if depths is None:
        depths = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    rows = []
    for d in depths:
        k = max(1, int(round(d * n)))
        targeted = y_sorted[:k]
        churners_caught = int(targeted.sum())
        precision = churners_caught / k
        expected_saves = churners_caught * save_rate
        value_saved = expected_saves * annual_value
        cost = k * offer_cost
        net = value_saved - cost
        rows.append({
            "target_depth": d,
            "customers_targeted": k,
            "churners_in_target": churners_caught,
            "precision": round(precision, 4),
            "expected_retained": round(expected_saves, 1),
            "gross_value_saved": round(value_saved, 0),
            "campaign_cost": round(cost, 0),
            "net_value": round(net, 0),
            "roi": round(net / cost, 2) if cost else 0.0,
        })
    return pd.DataFrame(rows)


def random_baseline(y_true, depth, annual_value=1200.0, offer_cost=40.0,
                    save_rate=0.30):
    """What a non-targeted (random) call list of the same size would return —
    the honest A/B counterfactual for the model-driven list."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    k = max(1, int(round(depth * n)))
    base_rate = y_true.mean()
    expected_churners = k * base_rate
    net = expected_churners * save_rate * annual_value - k * offer_cost
    return round(float(net), 0)


def uplift_vs_random(proba, y_true, depth=0.10, **kw):
    model_net = expected_value_curve(proba, y_true, depths=[depth], **kw)\
        .iloc[0]["net_value"]
    rand_net = random_baseline(y_true, depth, **kw)
    return {
        "target_depth": depth,
        "model_net_value": float(model_net),
        "random_net_value": float(rand_net),
        "uplift": round(float(model_net - rand_net), 0),
    }
