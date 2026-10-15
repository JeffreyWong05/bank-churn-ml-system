"""
Sanity tests — the model behaves like a probabilistic classifier and clears a
performance floor. A floor (not an exact value) catches a broken retrain
without being brittle to small run-to-run changes.
"""
from sklearn.metrics import roc_auc_score


def test_predicted_probabilities_in_unit_interval(model_bundle, prepared):
    proba = model_bundle["model"].predict_proba(prepared["X_test"])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_auc_clears_floor(model_bundle, prepared):
    proba = model_bundle["model"].predict_proba(prepared["X_test"])[:, 1]
    auc = roc_auc_score(prepared["y_test"], proba)
    assert auc > 0.75, f"AUC {auc:.3f} below floor — model may be broken"



def test_predicts_both_classes(model_bundle, prepared):
    preds = model_bundle["model"].predict(prepared["X_test"]) #get the predictions from the calibrated model

    assert set(preds) == {0,1}, f"model only predicted {set{preds}}. Only a single class was choosen."
    