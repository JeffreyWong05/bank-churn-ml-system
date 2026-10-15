"""
Schema tests — guard the shape of the data after prepare().

These are the cheapest, highest-value tests: they pin the assumptions every
later stage relies on. The column-name test below would have caught the real
HasCrCard / HasCreditCard rename bug immediately.
"""


def test_expected_feature_columns_present(prepared):
    expected = {"CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary"}
    missing = expected - set(prepared["X_train"].columns)
    assert not missing, f"missing expected columns: {missing}"


def test_no_identifier_leakage(prepared):
    # IDs must never reach the model — they invite memorisation / leakage.
    for col in ("CustomerId", "Surname"):
        assert col not in prepared["X_train"].columns


def test_target_is_binary(prepared):
    assert set(prepared["y_train"].unique()) <= {0, 1}


def test_stratified_split_preserves_churn_rate(prepared):
    gap = abs(prepared["y_train"].mean() - prepared["y_test"].mean())
    assert gap < 0.02, f"train/test churn rates diverged by {gap:.3f}"


def test_no_train_test_overlap(prepared):
    overlap = set(prepared["X_train"].index) & set(prepared["X_test"].index)
    assert not overlap, "rows appear in both train and test"
