"""
Contract tests — the report files contain exactly the keys the Streamlit app
indexes. The app does METRICS["roc_auc"], block["groups"], MONITORING["batches"].
If the pipeline ever stops producing those keys, the app crashes at load. These
tests turn that runtime crash into a caught failure before deploy.
"""
import os

from conftest import load_json


def test_metrics_has_keys_the_app_reads(reports_dir):
    m = load_json(os.path.join(reports_dir, "metrics.json"))
    assert {"roc_auc", "pr_auc", "brier"} <= m.keys()


def test_fairness_report_structure(reports_dir):
    f = load_json(os.path.join(reports_dir, "fairness_report.json"))
    assert f, "fairness report is empty"
    for attr, block in f.items():
        assert "groups" in block
        assert "demographic_parity_difference" in block
        assert "equal_opportunity_difference" in block


def test_monitoring_has_batches(reports_dir):
    mon = load_json(os.path.join(reports_dir, "monitoring.json"))
    assert "batches" in mon and len(mon["batches"]) > 0
    first = mon["batches"][0]
    assert {"batch", "auc", "max_psi"} <= first.keys()
