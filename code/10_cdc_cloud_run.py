"""
10_cdc_cloud_run.py -- the CDC half of notebook 10 (section 2: PC-G2 and LiM bootstraps) as a script,
so it can run on a many-core machine. NHANES is not touched.

Same settings as 10_sensitivity_algorithms.ipynb: SEED, BOOT = 50, MAX_N = 20000 discovery sample,
PC_ALPHA = 0.01, N_BINS = 5, and the same resample index sequence. The functions discovery_sample,
adj_pc and adj_lim are copied from the notebook unchanged.

Only difference in mechanics: each finished resample is saved as its own small file (parts/), so the
run can resume after any interruption and all 50 resamples can run at once. The final count is the sum
of the 50 adjacency matrices, exactly as in the notebook.

Input : cdcfull_train.csv (written by notebook 06)
Output: 10_edges_PC-G2_cdc.csv, 10_edges_LiM_cdc.csv, 10_checkpoint_*_cdc.npz  -> copy into results/

Examples
  python 10_cdc_cloud_run.py --train cdcfull_train.csv --out out --time-one          # time one resample
  python 10_cdc_cloud_run.py --train cdcfull_train.csv --out out --n-jobs 50         # the full run
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one thread per worker; the workers supply the parallelism

import argparse, contextlib, io, json, platform, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- settings: identical to notebook 10 ---------------------------------------------------------
SEED = 20261020
BOOT = 50
MAX_N = 20000
LIM_MAX_N = MAX_N
PC_MAX_N = MAX_N
PC_ALPHA = 0.01
N_BINS = 5
LABEL = "diabetes_any"
FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
            "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]   # li.CDC_DIABETES.numeric


# ---- copied from notebook 10, section 2 ---------------------------------------------------------
def discovery_sample(train):
    cols = [LABEL] + FEATURES
    x = train[cols].astype(float)
    return (x.sample(MAX_N, random_state=SEED) if len(x) > MAX_N else x).reset_index(drop=True), cols


def adj_pc(x):
    from causallearn.search.ConstraintBased.PC import pc
    xd = x.iloc[:PC_MAX_N].copy()
    for c in xd.columns:
        if xd[c].nunique() > N_BINS:
            xd[c] = pd.qcut(xd[c].rank(method="first"), N_BINS, labels=False)
        else:
            xd[c] = pd.factorize(xd[c], sort=True)[0]
    G = pc(xd.to_numpy(float), PC_ALPHA, "gsq", show_progress=False).G.graph
    return (G != 0) | (G.T != 0)


def adj_lim(x):
    import lingam
    x = x.iloc[:LIM_MAX_N].copy()
    binary = np.array([x[c].nunique() <= 2 for c in x.columns])
    for c, b in zip(x.columns, binary):
        x[c] = (x[c] == x[c].max()).astype(float) if b else (x[c] - x[c].mean()) / (x[c].std() or 1)
    m = lingam.LiM()
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(x.to_numpy(float), np.array([[0 if b else 1 for b in binary]]), only_global=True)
    A = np.abs(np.asarray(m.adjacency_matrix_)) > 0
    return A | A.T


ALGS = {"PC-G2": adj_pc, "LiM": adj_lim}


# ---- run mechanics -------------------------------------------------------------------------------
def one_resample(alg, x, idx, b, parts):
    path = parts / f"{alg}_{b:02d}.npy"
    if path.exists():
        return b, 0.0
    t0 = time.time()
    A = ALGS[alg](x.iloc[idx].reset_index(drop=True))
    tmp = parts / f"{alg}_{b:02d}.tmp.npy"
    np.save(tmp, np.asarray(A, dtype=bool)); os.replace(tmp, path)
    secs = time.time() - t0
    with open(parts.parent / "progress.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {alg} resample {b:02d} done in {secs / 60:.1f} min\n")
    return b, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--algs", default="PC-G2,LiM")
    ap.add_argument("--time-one", action="store_true", help="run PC-G2 resample 0 only and report the time")
    ap.add_argument("--smoke-rows", type=int, default=0, help="testing only: cap rows (results NOT for the paper)")
    ap.add_argument("--smoke-boot", type=int, default=0, help="testing only: number of resamples")
    a = ap.parse_args()

    global PC_MAX_N, LIM_MAX_N
    boot = BOOT
    if a.smoke_rows:
        PC_MAX_N = LIM_MAX_N = a.smoke_rows
    if a.smoke_boot:
        boot = a.smoke_boot
    out = Path(a.out); parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(a.train)
    x, cols = discovery_sample(train)
    print(f"train {len(train):,} rows | discovery sample {len(x):,} rows x {len(cols)} columns | "
          f"recorded rate {train[LABEL].mean():.4f}", flush=True)

    import causallearn, lingam, joblib, scipy, sklearn
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "n_jobs": a.n_jobs, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
           "joblib": joblib.__version__, "lingam": getattr(lingam, "__version__", "?"),
           "causal-learn": getattr(causallearn, "__version__", "?"),
           "train_rows": int(len(train)), "seed": SEED, "boot": boot, "max_n": MAX_N,
           "smoke_test": bool(a.smoke_rows or a.smoke_boot)}
    (out / "run_environment.json").write_text(json.dumps(env, indent=2))

    rng = np.random.default_rng(SEED)                      # same resample sequence as the notebook
    idxs = [rng.integers(0, len(x), len(x)) for _ in range(BOOT)][:boot]

    if a.time_one:
        b, secs = one_resample("PC-G2", x, idxs[0], 0, parts)
        print(f"PC-G2 resample 0: {secs / 60:.1f} min on one core", flush=True)
        (out / "time_one.txt").write_text(f"{secs:.0f} seconds\n")
        return

    for alg in a.algs.split(","):
        t0 = time.time()
        todo = [b for b in range(boot) if not (parts / f"{alg}_{b:02d}.npy").exists()]
        print(f"cdc {alg}: {boot - len(todo)} resamples already done, {len(todo)} to run on {a.n_jobs} cores", flush=True)
        Parallel(n_jobs=a.n_jobs)(delayed(one_resample)(alg, x, idxs[b], b, parts) for b in todo)
        count = np.zeros((len(cols), len(cols)))
        for b in range(boot):
            count = count + np.load(parts / f"{alg}_{b:02d}.npy")
        np.savez(out / f"10_checkpoint_{alg}_cdc.npz", count=count, done=boot)
        prob = count / boot
        e = pd.DataFrame([{"a": cols[i], "b": cols[j], "prob_adjacent": prob[i, j]}
                          for i in range(len(cols)) for j in range(i + 1, len(cols)) if prob[i, j] > 0])
        e.to_csv(out / f"10_edges_{alg}_cdc.csv", index=False)
        print(f"cdc {alg}: {boot}/{boot} resamples, {(time.time() - t0) / 60:.0f} min -> 10_edges_{alg}_cdc.csv "
              f"({len(e)} rows)", flush=True)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
