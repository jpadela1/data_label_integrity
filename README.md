# Label-integrity study (TRUST 2027): how to run

## Paper pipeline (current scope, since Oct 20, 2026)

The TRUST 2027 paper ("Invisible in Aggregate") uses **only** the notebooks in this table. Run them in this order.

| # | File | What it does | Time | Writes |
|---|---|---|---|---|
| 06 | `06_risk_group_discovery_cdc.ipynb` | CDC BRFSS: causal discovery, candidate groups, O/E check, baselines | ~10 min | `data/cdcfull_*.csv`, `results/06_*` |
| 07 | `07_nhanes_prepare.ipynb` | NHANES 2005-2018: download, clean, split | a few min | `data/nhanes_*.csv` |
| 08 | `08_risk_group_discovery_nhanes.ipynb` | NHANES: same analysis as 06, plus the blood-test ground truth (sections 9-10) | a few min | `results/08_*` |
| 10b | `10b_sensitivity_algorithms.ipynb` | Sensitivity run at **reduced size** (5,000 rows per resample). Run first, on Oct 21; kept and reported for disclosure | minutes | `results/10b_*` |
| 10 | `10_sensitivity_algorithms.ipynb` | Sensitivity run at **full size** (PC with G2 test, LiM). Loads saved edges if present | minutes if edges exist | `results/10_*` |
| - | `10_cdc_cloud_run.py` | The CDC half of notebook 10 as a script, for a many-core machine (see below) | 3 h 52 min on 64 cores | `10_edges_*_cdc.csv`, `10_checkpoint_*_cdc.npz` |
| 09 | `09_paper_figures_discovery.ipynb` | Paper figures, tables and `disc_macros.tex`. Run last | ~1 min | `figures/`, table `.tex` files |

Notebooks 06 and 08 are the same code with a `DATASET` switch. Notebooks 10 and 10b are the same analysis at two
sample sizes; both are reported in the paper's sensitivity table, so both are kept.

### Nov 11, 2026: CDC rerun on all 253,680 respondents

The first CDC analysis dropped rows that were exact copies of another row (253,680 -> 229,474). Those rows are not
duplicate records: the public file has no respondent identifier (personal information was removed), so identical rows
are different respondents who gave the same common answers, mostly healthy adults. Dropping them removed almost only
people without diabetes (rows with a twin: 1.3% positive; unique rows: 16.0%) and raised the recorded rate from 13.9%
to 15.3%. The CDC analysis was therefore rerun with all respondents kept:

* notebook 06: `df = raw.copy()` (was `raw.drop_duplicates()`), `REUSE_STAGE0 = False`;
* the full-size PC-G2 bootstrap, which took about 53 hours on a laptop for the first analysis, was rerun on AWS EC2
  (c7a.16xlarge, 64 cores, us-east-1) with `10_cdc_cloud_run.py`, so all 50 resamples ran at once, each on its own
  core. It finished in 3 h 52 min and cost about $14. The script uses the same seed, discovery sample and resample
  sequence as notebook 10; run on the earlier data at 5,000 rows it reproduced `10b_edges_PC-G2_cdc.csv` exactly.
  Machine and package versions: `results/10_cdc_cloud_run_environment.json`; timings: `results/10_cdc_cloud_run_progress.log`.
* The results of the first (de-duplicated) analysis are kept in `backup_20261111_dedupe/`.

Other notes:

* CDC label: `Diabetes_binary` is **diabetes only** (35,346 positives = class 2 of the three-class file; the 4,631
  prediabetes respondents are coded 0). `data/diabetes_012_health_indicators_BRFSS2015.csv/` was downloaded from
  Kaggle only to verify this; no notebook uses it.
* Notebook 08, section 10 (added Nov 11): short lists for clustering and the protected-only audit, and two ablations
  of the method (`results/08_ground_truth_shortlist_extra.csv`, `results/08_cluster_profiles.csv`).

## Earlier pipeline (label injection and repair; thesis chapter, not in the TRUST paper)

Notebooks 00-05b below belong to the earlier scope of the study. They are kept for the dissertation chapter.

Run the notebooks **in order**, top to bottom, each one completely before the next.
Every notebook starts with "What this step does and why" and ends with "What to look for".

| # | Notebook | Time on a laptop CPU | Writes |
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
| NHANES | `nhanes*.csv` | downloaded by notebook 07 |

Older files kept as fallbacks: `diabetes_130.csv` is used only if the raw Diabetes download
fails (it has no patient IDs, so repeat patients cannot be removed). `german_credit.csv` is the
old Statlog version and is **not used**.

## One-time setup (PowerShell, from the `safety_notebooks` folder)

```
.\.venv312\Scripts\activate
pip install cleanlab xgboost
```

`.venv312` already has causal-learn, lingam, scikit-learn, pandas, ucimlrepo and Jupyter.
Choose the `.venv312` kernel when you open the notebooks.

## Files

- `li_lib.py`: all shared functions, with the experiment grid (doses, mechanisms, seeds, review budget, audit sizes) at the top. Change settings there, not in the notebooks.
- `stage0/stage0_diagnose.py`: your September Stage 0 script, copied unchanged from `Archive/stage0/`.
- `pilot_standin_20261009/`: trial run on stand-in data. **Not for the paper.**
