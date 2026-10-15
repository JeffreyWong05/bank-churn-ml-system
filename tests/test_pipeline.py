"""
Smoke test — run the pipeline end to end into a temp dir and assert every file
the dashboard reads actually gets produced, in the right folder.

This is the test that would have caught the metrics.json-in-the-wrong-folder
bug: it asserts metrics.json lands in reports/, not models/.

`tmp_path` is a pytest built-in giving the test a clean temp directory.
`monkeypatch` lets us repoint the pipeline's output paths at that temp dir so
the test never touches the committed artifacts.
"""


def test_pipeline_produces_app_inputs(tmp_path, monkeypatch):
    import run_pipeline

    monkeypatch.setattr(run_pipeline, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(run_pipeline, "REPORTS", str(tmp_path / "reports"))

    run_pipeline.main(n_trials=3)   # few Optuna trials => fast

    reports = tmp_path / "reports"
    for fname in ["metrics.json", "fairness_report.json",
                  "monitoring.json", "shap_summary.png"]:
        assert (reports / fname).exists(), f"missing app input: {fname}"

    # Regression guard: metrics.json belongs in reports/, NOT models/.
    assert not (tmp_path / "models" / "metrics.json").exists()
    assert (tmp_path / "models" / "model.joblib").exists()
