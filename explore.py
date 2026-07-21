"""
explore.py — a no-setup tour of the dataset.

Run this once to build intuition before you touch the model:

    python explore.py

It prints the handful of numbers you'll actually talk about in an interview
(who churns, and by how much), so you don't have to write any pandas yourself.
"""
import os
import pandas as pd

CSV = os.path.join("data", "churn.csv")


def load():
    df = pd.read_csv(CSV)
    # The raw file has spaces in some headers ("Num Of Products"); normalise.
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    return df


def rate_by(df, col):
    """Churn rate within each group of `col`, sorted worst-first."""
    out = (df.groupby(col)["Churn"]
             .agg(customers="count", churn_rate="mean")
             .sort_values("churn_rate", ascending=False))
    out["churn_rate"] = (out["churn_rate"] * 100).round(1)
    return out


def main():
    df = load()

    print("=" * 60)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Overall churn rate: {df['Churn'].mean() * 100:.1f}%")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print("=" * 60)

    print("\nChurn rate by GEOGRAPHY (fairness slice #1):")
    print(rate_by(df, "Geography").to_string())

    print("\nChurn rate by GENDER (fairness slice #2):")
    print(rate_by(df, "Gender").to_string())

    print("\nChurn rate by ACTIVE MEMBER (0 = inactive):")
    print(rate_by(df, "IsActiveMember").to_string())

    print("\nChurn rate by NUMBER OF PRODUCTS:")
    print(rate_by(df, "NumOfProducts").to_string())

    print("\nAge — churners vs stayers (mean):")
    print(df.groupby("Churn")["Age"].mean().round(1).to_string())

    print("\nBalance — churners vs stayers (mean):")
    print(df.groupby("Churn")["Balance"].mean().round(0).to_string())

    print("\n" + "=" * 60)
    print("What to notice (your interview talking points):")
    print("- ~1 in 5 customers churn; the classes are imbalanced, so you")
    print("  evaluate with AUC/PR-AUC, not accuracy.")
    print("- Germany churns far more than France/Spain — this is the base-rate")
    print("  gap that later shows up in the fairness audit.")
    print("- Inactive members and customers with 3-4 products churn much more.")
    print("- Churners skew older with higher balances — real, explainable signal.")
    print("=" * 60)


if __name__ == "__main__":
    main()
