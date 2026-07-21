"""
Train a LightGBM churn classifier with Optuna hyperparameter search and
probability calibration, then persist the model and headline metrics.

Design choices worth noting for a reviewer:
  * We tune against ROC-AUC on a validation fold, not accuracy: the classes
    are imbalanced (~20% churn), so accuracy is misleading.
  * Probabilities are calibrated (isotonic) because the business layer prices
    a retention campaign off predicted probabilities, and mis-calibrated
    scores would distort the expected-value ranking.
  * Everything is seeded so the pipeline is reproducible.
"""
from __future__ import annotations

import json
import os
import joblib
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42


def _objective(trial, X, y, cat_features):
    params = {
        "objective": "binary",
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 90),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    # LightGBM auto-detects pandas 'category' dtype columns as categorical.
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()


def train(data: dict, n_trials: int = 25, out_dir: str = "models"):
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]
    cat = data["categorical"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: _objective(t, X_tr, y_tr, cat),
                   n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"objective": "binary", "random_state": SEED,
                 "n_jobs": -1, "verbose": -1})

    base = LGBMClassifier(**best)
    base.fit(X_tr, y_tr)

    # Calibrate on held-out folds so business probabilities are trustworthy.
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)

    proba = calibrated.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "pr_auc": round(average_precision_score(y_te, proba), 4),
        "recall": round(recall_score(y_te, preds), 4),
        "precision": round(precision_score(y_te, preds), 4),
        "f1": round(f1_score(y_te, preds), 4),
        "brier": round(brier_score_loss(y_te, proba), 4),
        "cv_auc_mean": round(study.best_value, 4),
        "n_trials": n_trials,
        "best_params": {k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in study.best_params.items()},
    }

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({"model": calibrated, "base": base,
                 "features": data["feature_names"], "categorical": cat},
                os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return calibrated, base, metrics


if __name__ == "__main__":
    from data import prepare
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = prepare(os.path.join(here, "data", "churn.csv"))
    _, _, m = train(d, n_trials=25, out_dir=os.path.join(here, "models"))
    print(json.dumps(m, indent=2))
