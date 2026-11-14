# Label-integrity study (TRUST 2027): how to run

Last updated Nov 14, 2026.

## Paper pipeline

The TRUST 2027 paper ("Invisible in Aggregate") uses **only** the files in this table. Run them in this order.

| # | File | What it does | Time | Writes |
|---|---|---|---|---|
| 06 | `06_risk_group_discovery_cdc.ipynb` | CDC BRFSS (253,680 respondents): causal discovery, candidate groups, observed-versus-expected (O/E) check, comparison methods | ~10 min | `data/cdcfull_*.csv`, `results/06_*` |
| 07 | `07_nhanes_prepare.ipynb` | NHANES 2005-2018: download, clean, split | a few min | `data/nhanes_*.csv` |
| 08 | `08_risk_group_discovery_nhanes.ipynb` | NHANES: same analysis as 06, plus the blood-test ground truth (section 9), the extra short lists and ablations (section 10) and per-record rates (section 10b) | a few min | `results/08_*` |
| 10b | `10b_sensitivity_algorithms.ipynb` | Sensitivity analysis at **reduced size** (5,000 rows per resample). Run first; kept and reported for disclosure | minutes | `results/10b_*` |
| 10 | `10_sensitivity_algorithms.ipynb` | Sensitivity analysis at **full size** (PC with the G2 test, and LiM). **Loads the saved edge files if they are present** | minutes with the saved files; many hours without (see below) | `results/10_*` |
| - | `10_cdc_cloud_run.py` | The CDC half of notebook 10, section 2, as a script for a many-core machine | 3 h 52 min on 64 cores | `10_edges_*_cdc.csv`, `10_checkpoint_*_cdc.npz` |
| 09 | `09_paper_figures_discovery.ipynb` | Every paper figure and table, and `disc_macros.tex` (every number quoted in the text). Run last | ~1 min | `figures/paper_disc/` |

Notebooks 06 and 08 are the same code with a `DATASET` switch. Notebooks 10 and 10b are the same analysis at two
sample sizes; both are reported in the paper's sensitivity table, so both are kept.

## The one slow step, and how to avoid waiting for it

**Which step.** Notebook 10, **section 2 ("Bootstrap PC-G2 and LiM on the same resamples Stage 0 used")**, for the
**CDC dataset with the PC-G2 algorithm**. It runs the PC algorithm 50 times, once per bootstrap resample, each on
20,000 rows and 22 variables. Everything else in the pipeline, including LiM (under a minute per resample) and the
NHANES half of the same section, is fast.

**How long it takes.** On a laptop this one step took about **53 hours**. On a 64-core cloud machine it took
**3 hours 52 minutes**. Measured on that machine: one resample takes between 54 and 232 minutes (median about 115),
and the 50 resamples add up to about 100 core-hours. So the wall-clock time is roughly 100 hours divided by the number
of cores you give it, and never less than the slowest resample (about 4 hours).

### Option 1 (recommended): do not rerun it

The finished results of this step are in the repository:

* `results/10_edges_PC-G2_cdc.csv`, `results/10_edges_LiM_cdc.csv` (and the same two files for `nhanes`)
* `results/10_checkpoint_*.npz` (the running totals behind them)

Notebook 10, section 2, has `REUSE = True`. With that setting it **loads these files and skips the bootstrap**, and
the whole notebook finishes in minutes. Leave `REUSE = True` unless you intend to recompute.

### Option 2: rerun it on a many-core machine with `10_cdc_cloud_run.py`

The script is section 2 of notebook 10 for the CDC dataset, with the same seed (`20261020`), the same 20,000-row
discovery sample, the same 50 resamples in the same order, and the functions copied unchanged. As a check, run on the
earlier data at 5,000 rows it reproduced `results/10b_edges_PC-G2_cdc.csv` exactly.

Configuration we used (Nov 11, 2026):

| Setting | Value |
|---|---|
| Machine | AWS EC2 `c7a.16xlarge`: 64 vCPUs, 128 GiB memory, region us-east-1, on-demand |
| Operating system | Ubuntu 24.04 (Linux, glibc 2.39), Python 3.12.3 |
| Parallel workers | `--n-jobs 50`: all 50 resamples at once, one resample per core (joblib) |
| Threads per worker | 1 (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` are set to 1 by the script, so the workers do not compete for cores) |
| Packages | numpy 2.2.6, pandas 3.0.6, scipy 1.13.1, joblib 1.6.0, lingam 1.13.0, causal-learn (version in `requirements.txt`) |
| Input | `data/cdcfull_train.csv` (152,208 rows; written by notebook 06) |
| Run time and cost | 3 h 52 min; about $14 including test launches |

The exact machine and package record is `results/10_cdc_cloud_run_environment.json`; the time of every resample is in
`results/10_cdc_cloud_run_progress.log`.

Steps:

```
pip install numpy pandas scipy scikit-learn joblib lingam causal-learn

# optional: time a single resample first, to estimate the full run on your machine
python 10_cdc_cloud_run.py --train cdcfull_train.csv --out out --time-one

# the full run (use a number of workers no larger than your core count)
nohup python 10_cdc_cloud_run.py --train cdcfull_train.csv --out out --n-jobs 50 > run.log 2>&1 &
```

When it finishes it writes `out/DONE`. Copy `out/10_edges_PC-G2_cdc.csv`, `out/10_edges_LiM_cdc.csv` and the two
`out/10_checkpoint_*_cdc.npz` files into `results/`, then run notebook 10 (it loads them) and notebook 09.
Remember to stop or terminate the cloud machine afterwards.

### Option 3: rerun it inside notebook 10 on your own computer

Set `REUSE = False` in section 2, or delete the `10_edges_*` files you want recomputed. The notebook runs the resamples
in parallel on all but two of your processor cores (`N_JOBS` in `li_lib.py`). Expect roughly 100 hours divided by
your core count for the CDC PC-G2 step.

### If the run stops, it resumes; you never start over

Both routes save their progress, so a stop of any kind (closed laptop, restart, lost connection, stopping it on
purpose) costs only the work that was in progress at that moment.

| Route | What is saved, and when | How to resume | What is lost |
|---|---|---|---|
| Notebook 10, section 2 | The running total, after each batch of `N_JOBS` resamples, in `results/10_checkpoint_{algorithm}_{dataset}.npz`. Each finished algorithm and dataset is also saved at once as `results/10_edges_{algorithm}_{dataset}.csv` | Run the section 2 cell again | Only the batch that was running |
| `10_cdc_cloud_run.py` | Each resample, the moment it finishes, as its own file `out/parts/{algorithm}_{NN}.npy` | Run the same command again with the same `--out` folder | Only the resamples that were running |

On restart the script prints, for example, `cdc PC-G2: 37 resamples already done, 13 to run on 50 cores`.

**To start over on purpose** (for example after changing the data): delete the matching `10_checkpoint_*.npz` and
`10_edges_*.csv` files in `results/` (notebook), or the `out/parts/` folder (script). Saved progress is not tied to a
version of the data, so **always delete it after the input data changes**; otherwise old and new resamples are mixed.

## What notebook 09 produces for the paper

Everything is written to `figures/paper_disc/`, and copied to the paper folder by the last cell when
`EXPORT_TO_PAPER = True`.

| File | In the paper |
|---|---|
| `pipeline_discovery.pdf` | Figure: the method pipeline |
| `stage0_dag_nhanes.pdf`, `stage0_dag_cdc.pdf` | Figure: what causal discovery found around the label and the declared variable |
| `shortlist_nhanes.pdf` | Figure: mean rate ratio of each method's short list |
| `tab_prf.tex` | Table: precision, recall and F1 of each short list against the answer key (section 4b) |
| `tab_answer_key.tex` | Table: the truly under-recorded groups (answer key) |
| `tab_nhanes_check.tex` | Table: the O/E check against the blood-test result |
| `tab_cdc_flags.tex` | Table: CDC flags and how the other methods rank the same groups |
| `tab_sensitivity.tex` | Table: the same rule with three causal-discovery algorithms |
| `disc_macros.tex` | Every number quoted in the text, as a `\DS...` macro |

Terms used in the paper and in these outputs:

* **Undiagnosed rate** (called "missed share" in older files and in column names such as `missed_share_group`): of the
  adults in a group who have diabetes, the percentage never told.
* **Rate ratio** (column `missed_ratio`): a group's undiagnosed rate divided by everyone else's.
* **Label-link filter** (called "graph rule" in older files): the O/E check is applied only to attributes linked to
  the declared labeling-process variable and not linked to the label.

**Changing a printed label in notebook 09.** Rows are looked up by their name in the results files, for example
`ablation: O/E without the graph rule`. Do not change those names. Change only the printed text: the right-hand side
of a `"name in file": "printed text"` pair, or the third item in the `PRF_ROWS` list in section 4b.

## Data notes

* **All CDC respondents are kept (since Nov 11, 2026).** The first CDC analysis dropped rows that were exact copies of
  another row (253,680 -> 229,474). Those rows are not duplicate records: the public file has no respondent identifier,
  so identical rows are different respondents who gave the same common answers, mostly healthy adults. Dropping them
  removed almost only people without diabetes (rows with a twin: 1.3% positive; unique rows: 16.0%) and raised the
  recorded rate from 13.9% to 15.3%. Notebook 06 now uses `df = raw.copy()` (was `raw.drop_duplicates()`). The results
  of the first analysis are kept in `backup_20261111_dedupe/`.
* **CDC label.** `Diabetes_binary` is **diabetes only** (35,346 positives = class 2 of the three-class file; the 4,631
  prediabetes respondents are coded 0). `data/diabetes_012_health_indicators_BRFSS2015.csv/` was downloaded from
  Kaggle only to verify this; no notebook uses it.
* **NHANES label.** Recorded label = "told by a doctor that you have diabetes" (`DIQ010 = 1`; "borderline" is coded 0).
  Ground truth = told, or measured HbA1c of 6.5% or higher. Rates are unweighted.
* **Environment.** The paper notebooks run from the `.venv` environment inside this folder; `requirements.txt` lists
  the packages. (The setup note further down, which mentions `.venv312`, belongs to the earlier pipeline.)

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
| South German Credit (UCI 573) | `SouthGermanCredit.asc` | downloaded by notebook 00 |

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
