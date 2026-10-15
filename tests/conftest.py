"""
Shared pytest fixtures.

A fixture is reusable setup that tests ask for by naming it as an argument.
Expensive work (loading data, loading the model) is done once and shared, so
the suite stays fast. `scope="session"` means "build this once for the whole
test run," not once per test.
"""
import json
import os
import sys

import joblib
import pytest

# Make the project's src/ importable as `data`, `model`, `run_pipeline`, etc.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data as data_mod  # noqa: E402


@pytest.fixture(scope="session")
def prepared():
    """The train/test split the whole project is built on."""
    return data_mod.prepare(os.path.join(ROOT, "data", "churn.csv"))


@pytest.fixture(scope="session")
def model_bundle():
    """The committed, deployed model artifact — the one the app actually loads."""
    return joblib.load(os.path.join(ROOT, "models", "model.joblib"))


@pytest.fixture(scope="session")
def reports_dir():
    """Path to the committed reports/ the dashboard reads at runtime."""
    return os.path.join(ROOT, "reports")


def load_json(path):
    with open(path) as f:
        return json.load(f)
