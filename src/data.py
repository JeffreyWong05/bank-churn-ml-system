"""
Data loading, cleaning, and feature engineering for the bank churn model.

The raw file ships with slightly inconsistent column names (spaces), so we
normalise them here and keep a single source of truth for the feature set.
Protected attributes (Gender, Geography) are preserved separately so the
fairness audit can slice on them without letting them leak into the model in
a way we can't account for.
"""
from __future__ import annotations

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_COLUMNS = {
    "Num Of Products": "NumOfProducts",
    "Has Credit Card": "HasCrCard",
    "Is Active Member": "IsActiveMember",
    "Estimated Salary": "EstimatedSalary",
}

TARGET = "Churn"
PROTECTED = ["Gender", "Geography"]
DROP = ["CustomerId", "Surname"]

# Categorical columns the model is allowed to use as features.
CATEGORICAL = ["Geography", "Gender"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=RAW_COLUMNS)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-motivated features a bank analyst would actually reach for."""
    df = df.copy()
    # Balance-to-salary ratio: liquidity signal relative to income.
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)
    # Products held per year of tenure: engagement velocity.
    df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1.0)
    # Zero-balance flag: customers who parked no money are a distinct segment.
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    # Coarse age bands used later for segmentation views.
    df["AgeBand"] = pd.cut(
        df["Age"], bins=[17, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"]
    ).astype(str)
    return df


def prepare(path: str, test_size: float = 0.2, seed: int = 42):
    """Return train/test splits plus the protected-attribute frames for audit."""
    df = engineer(load_raw(path))

    y = df[TARGET].astype(int)
    protected = df[PROTECTED].copy()
    X = df.drop(columns=[TARGET])

    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    X["AgeBand"] = X["AgeBand"].astype("category")

    X_tr, X_te, y_tr, y_te, prot_tr, prot_te = train_test_split(
        X, y, protected, test_size=test_size, random_state=seed, stratify=y
    )
    return {
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
        "prot_train": prot_tr, "prot_test": prot_te,
        "feature_names": list(X.columns),
        "categorical": [c for c in X.columns if str(X[c].dtype) == "category"],
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = prepare(os.path.join(here, "data", "churn.csv"))
    print("train:", d["X_train"].shape, "test:", d["X_test"].shape)
    print("features:", d["feature_names"])
