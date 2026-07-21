# Getting Started (Phase 0 + Phase 1)

A copy-paste setup guide. Commands are given for **Windows (PowerShell)** and
**macOS / Linux**. Pick your OS column and run them top to bottom.

> Rule of thumb: on Windows type `python`, on macOS/Linux type `python3`.

---

## Phase 0 — Environment & repo

### 0.1  Check Python (need 3.11 or 3.12)

**Windows**
```powershell
python --version
```
**macOS / Linux**
```bash
python3 --version
```
If it's missing or older than 3.11: install from https://www.python.org/downloads/
(on Windows, tick **"Add python.exe to PATH"** during install). Then re-open the
terminal and check again.

### 0.2  Open the project folder
Unzip the project, then `cd` into it so your prompt is inside `bank-churn-ml-system`:
```bash
cd path/to/bank-churn-ml-system
```

### 0.3  Create and activate a virtual environment (pin it to Python 3.11)
A venv keeps this project's packages separate from the rest of your system.
Important: **a venv locks in whatever Python version you use to create it.** So
you don't need to change your system default — just call 3.11 explicitly when
building the venv. The project folder and the Python version are independent.

**Windows (PowerShell)** — use the `py` launcher to pick 3.11:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version          # should say 3.11.x
```
> If you see *"running scripts is disabled on this system"*, run this once, then
> re-run the activate line:
> ```powershell
> Set-ExecutionPolicy -Scope Process RemoteSigned
> ```

**macOS / Linux / WSL2** — call `python3.11` explicitly:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version          # should say 3.11.x
```
> On Ubuntu/WSL2, if it says venv isn't available, install it once:
> `sudo apt install python3.11-venv`, then rerun.

Your prompt should now start with `(.venv)`. Once activated, `python` and `pip`
inside it are 3.11 regardless of the system default — which is exactly why the
global default doesn't matter here.

**Already made a `.venv` with the wrong version?** Delete and recreate it in the
same folder (no new folder needed):
```bash
# macOS / Linux / WSL2
rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate
```
```powershell
# Windows
Remove-Item -Recurse -Force .venv ; py -3.11 -m venv .venv ; .venv\Scripts\Activate.ps1
```

### 0.3b  (Optional) Changing your default Python — usually skip this
You don't need this for the project, and on Linux it can break OS tools, so
prefer the explicit `python3.11 -m venv` approach above. If you still want it:

- **Ubuntu / WSL2:** don't remap `/usr/bin/python3` (apt depends on it). Instead
  add a per-user alias to `~/.bashrc`: `alias python=python3.11`, then
  `source ~/.bashrc`. For a proper version manager, use `pyenv`.
- **Windows:** move your 3.11 folder above 3.10 in PATH (Environment Variables →
  Path), or set an environment variable `PY_PYTHON=3.11` so the `py` launcher
  defaults to 3.11 in new terminals.

### 0.4  macOS ONLY — install libomp (LightGBM needs it)
Skip on Windows/Linux. Without this, `import lightgbm` fails with a
"Library not loaded: libomp.dylib" error.
```bash
brew install libomp
```
(No Homebrew? Install it from https://brew.sh first.)

### 0.5  Install dependencies
```bash
pip install -r requirements.txt
```
This pulls in shap/numba and can take **3–6 minutes** — that's normal, not a hang.

### 0.6  Verify the install
```bash
python -c "import lightgbm, shap, streamlit, optuna; print('all good')"
```
Windows users: still `python`. You should see `all good`.

### 0.7  Put it on GitHub (public repo)
1. On github.com click **New repository** → name it `bank-churn-ml-system` →
   **Public** → do **not** add a README (this project already has one) → Create.
2. Back in the terminal:
```bash
git init
git add .
git commit -m "Initial commit: bank churn ML system"
git branch -M main
git remote add origin https://github.com/JeffreyWong05/bank-churn-ml-system.git
git push -u origin main
```
The included `.gitignore` keeps `.venv/` and caches out of the repo.

---

## Phase 1 — Data & first run

### 1.1  Confirm the dataset is there
```bash
python -c "import os; print('OK' if os.path.exists('data/churn.csv') else 'MISSING')"
```
If it says MISSING, download it (one line):

**macOS / Linux**
```bash
curl -L -o data/churn.csv "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Bank%20Churn%20Modelling.csv"
```
**Windows (PowerShell)** — note `curl.exe`, not `curl`:
```powershell
curl.exe -L -o data/churn.csv "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Bank%20Churn%20Modelling.csv"
```

### 1.2  Explore the data (no coding required)
```bash
python explore.py
```
Read the output. What to notice:
- **~20% churn** → imbalanced, so you evaluate with AUC/PR-AUC, not accuracy.
- **Germany churns ~32% vs ~16%** elsewhere → the base-rate gap that later
  drives the fairness finding.
- **Inactive members** and **3–4 product holders** churn far more.
- **Churners skew older with higher balances** → real, explainable signal.

### 1.3  Run the full pipeline (trains the model, writes all reports)
```bash
python src/run_pipeline.py
```
Takes ~1–2 minutes (Optuna runs 25 tuning trials). Expect **ROC-AUC ≈ 0.87**.
New files appear in `models/` and `reports/`.

### 1.4  Launch the dashboard locally
```bash
streamlit run app/streamlit_app.py
```
A browser tab opens at http://localhost:8501. Click through the tabs
(Score a customer → SHAP → Fairness → Monitoring → Business case). Press
`Ctrl+C` in the terminal to stop it.

If all of that worked, you're ready to extend it (see EXTENSIONS.md) and deploy.
