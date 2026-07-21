# Bank Customer Churn — An End-to-End Applied ML System

Predicting which bank customers are about to leave, **and** doing the parts that
separate a notebook from a production model: interpretability, a fairness audit,
drift monitoring, and a costed retention-campaign business case.

Built on 10,000 retail-banking customers. Reproducible from raw data with one
command, with an interactive dashboard.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-SUBDOMAIN.streamlit.app)

**🔗 Live dashboard:** https://YOUR-SUBDOMAIN.streamlit.app  _(replace after you deploy)_
**▶ Reproduce everything:** `python src/run_pipeline.py`

![Dashboard screenshot](docs/screenshot.png)
<!-- After deploying: take a screenshot of the dashboard, save it as docs/screenshot.png, and commit it. -->

---

## Why this project

Most churn projects stop at "I trained a model and got 0.86 AUC." In a regulated
bank, that's the easy part. This project covers the full lifecycle an applied ML
scientist owns:

| Lifecycle stage | What's here |
|---|---|
| Framing | Churn reframed as a *retention-targeting* decision with a dollar value |
| Modelling | LightGBM + Optuna search, probability **calibration** |
| Interpretability | SHAP global + per-customer local explanations |
| Fairness | Group audit across **Gender** and **Geography** |
| Monitoring | PSI drift detection + rolling AUC on a simulated stream |
| Business impact | Expected-value curve by targeting depth + uplift vs random |

## Headline results

- **ROC-AUC 0.868**, PR-AUC 0.719, Brier 0.099 (calibrated) on held-out data.
- **Top churn drivers** (SHAP): number of products, age, active-membership status.
- **Fairness finding:** strong recall parity across gender (gap 0.06) but a wider
  0.26 gap across geography, tracking Germany's higher base churn rate (32% vs
  ~16%) — the kind of disparity a lender must surface and document before shipping.
- **Monitoring:** injected drift drops batch AUC ~0.87 → 0.78 and trips the PSI
  alarm on `Balance`.
- **Business case:** targeting the top 10% highest-risk customers yields ~**$46K**
  more net value than a random call list of the same size.

## Quick start

```bash
pip install -r requirements.txt
python explore.py            # tour the data
python src/run_pipeline.py   # train + generate every report
streamlit run app/streamlit_app.py
```

New here? See **GETTING_STARTED.md** for a copy-paste, OS-by-OS walkthrough.
Want to extend it? See **EXTENSIONS.md**.

## Deploy (free)

Push to a public GitHub repo, then go to https://share.streamlit.io, connect
GitHub, and point it at `app/streamlit_app.py`. Free tier, no card required.

## Honest limitations

- Public benchmark dataset, not real customer data — illustrative of method.
- The retention **save rate** is an assumption; in production it should be
  learned from an A/B holdout (see EXTENSIONS.md #4).
- Drift is *simulated* to demonstrate the monitoring logic.

## Tech

Python · LightGBM · Optuna · scikit-learn · SHAP · pandas/NumPy · Streamlit
