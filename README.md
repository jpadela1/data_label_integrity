# Label-integrity study (TRUST 2027): how to run

Run the notebooks **in order**, top to bottom, each one completely before the next.
Every notebook starts with "What this step does and why" and ends with "What to look for".

| # | Notebook | Time on a laptop CPU | Writes/Outputs |
|---|---|---|---|
| 00 | prepare_data | 2 min | `data/*_train/val/test.csv` |
| 01 | stage0_diagnose | 20-60 min | `stage0_out/`, frozen risk strata |
| 02 | inject_noise | 1 min | `results/noisy/` |
| 03 | detect | 20-40 min | `results/detect/`, `03_detect_summary.csv` |
| 03b | localize | 5-10 min | `03b_localize.csv` |
| 04 | repair_verify | 2-3 hours | `04_verify.csv` |
| 05 | figures_tables | 1 min | `figures/` |

No GPU is needed. Each long notebook has a `QUICK = True` switch for a fast trial run
(fewer seeds or bootstraps). Never use QUICK results in the paper.

## Datasets

| Dataset | File in `data/` | How it gets there |
|---|---|---|
| Diabetes 130 (UCI 296) | `diabetic_data.csv` (raw, with patient IDs) | downloaded by notebook 00 |
| CDC Diabetes Health Indicators (UCI 891) | `cdc_diabetes_raw.csv` | downloaded by notebook 00 with `ucimlrepo` |
| Taiwan Credit (UCI 350) | `taiwancredit_raw.csv` | downloaded by notebook 00 |

Older files kept as fallbacks: `diabetes_130.csv` is used only if the raw Diabetes download
fails (it has no patient IDs, so repeat patients cannot be removed). `german_credit.csv` and `southgerman.asc` are the
old / small versions and are **not used**.

## One-time setup (PowerShell, from the `safety_notebooks` folder)

```
.\.venv312\Scripts\activate
pip install cleanlab xgboost
```

`.venv312` already has causal-learn, lingam, scikit-learn, pandas, ucimlrepo and Jupyter.
Choose the `.venv312` kernel when you open the notebooks.

## Files

- `li_lib.py`: all shared functions, with the experiment grid (doses, mechanisms, seeds, review budget, audit sizes) at the top. Change settings there, not in the notebooks.
- `stage0/stage0_diagnose.py`: old September Step 0 script, copied unchanged from `Archive/stage0/`.
