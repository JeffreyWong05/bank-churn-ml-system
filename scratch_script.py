import sys
sys.path.insert(0, "src")          # same line conftest.py uses
import joblib
import data

prepared = data.prepare("data/churn.csv")
bundle = joblib.load("models/model.joblib")

print("=== prepared ===")
print(type(prepared))                          # dict
print(prepared.keys())                          # what keys exist
print("X_train shape:", prepared["X_train"].shape)
print("columns:", prepared["X_train"].columns.tolist())
print(prepared["X_train"].head())               # first 5 rows
print("class balance:\n", prepared["y_train"].value_counts())

print("\n=== model_bundle ===")
print(type(bundle))                             # dict
print(bundle.keys())                            # dict_keys(['model','base','features','categorical'])
print("model type:", type(bundle["model"]))
print("features:", bundle["features"])