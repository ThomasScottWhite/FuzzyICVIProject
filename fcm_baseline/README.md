# Independent FCM / K-means Baseline for the DMSC Silhouette-Gate Study

Clusters the four datasets from the DMSC experiments (**Iris, Wine, Covertype, Fashion-MNIST**)
with plain Fuzzy C-Means and K-means: **no autoencoder, no silhouette gate, no guidance**.
It answers: *"What does clustering achieve without any learned representation?"* — the
reference point needed to judge whether DMSC (with or without guidance) adds anything.

## What gets run

| Axis | Values | Notes |
|---|---|---|
| Datasets | `iris`, `wine`, `covertype`, `fashion_mnist` | Preprocessing identical to paper §VI-B (min-max / ÷255; Covertype's 44 binary columns untouched) |
| Methods | `fcm`, `kmeans` | FCM is our own implementation (m=2, ≤50 iters, tol 1e-4 = DMSC settings). K-means is scikit-learn (k-means++, 10 restarts) |
| Spaces | `raw`, `pca3` | `raw` = original features. `pca3` = 3-D PCA, matching DMSC's 3-D latent size but with *no learning* |
| Seeds | 42–46 | Same as Experiment 1 |
| Metrics | ACC (Hungarian), NMI, ARI on full data; exact silhouette on the fixed subset | Subset: all points (Iris/Wine), 5,000 points seeded 2026 (large sets) |

k equals the number of true classes (benchmark convention, as in the paper). Labels are used
only for scoring. Final hard assignments are saved for every run (`results/labels/*.npy`),
so partitions can be compared directly (e.g. `adjusted_rand_score(a, b)`).

## Layout

```
fcm_baseline/            Python package
  data.py                loading + preprocessing + fixed evaluation subset
  fcm.py                 chunked, from-scratch Fuzzy C-Means
  metrics.py             ACC / NMI / ARI / reference silhouette
  run_baseline.py        main CLI (resumable)
  aggregate.py           builds summary.csv / summary.md
  prepare_data.py        one-time dataset download/cache
slurm/
  setup_env.sh           create venv + install deps (login node)
  run_baseline.sbatch    job array, one task per dataset
  aggregate.sbatch       merge results
  submit_all.sh          cache data → submit array → submit aggregation (dependent)
tests/test_baseline.py   unit tests (no internet needed)
requirements.txt
```

## Running on the HPC cluster (Slurm)

```bash
# 1. Copy the folder to the cluster, then from its root on the LOGIN node:
bash slurm/setup_env.sh          # once; edit the `module load python` line if needed
bash slurm/submit_all.sh         # downloads data, submits 4-task array + aggregation job
squeue -u $USER                  # monitor;  logs/ holds stdout/stderr
cat results/summary.md           # when finished
```

Before first use, open `slurm/*.sbatch` and uncomment/edit the `--partition` / `--account`
lines if Mill requires them. The jobs are CPU-only (no GPU request), 4 cores / 16 GB / 4 h per
dataset; adjust if needed. Jobs are resumable: resubmitting skips finished runs.

Compute nodes are usually offline, which is why `prepare_data` runs on the login node.

## Running locally / quick checks

```bash
pip install -r requirements.txt
pytest -q                                              # 11 tests
python -m fcm_baseline.run_baseline --datasets iris wine --output-dir results
python -m fcm_baseline.aggregate --results-dir results
```

Useful flags: `--methods fcm`, `--spaces raw`, `--init random`, `--seeds 42`,
`--fuzzifier`, `--max-iter`, `--tol`, `--subset-indices file.npy`, `--no-resume`.

## Output

* `results/results_<dataset>.csv` — one row per run (scores, iterations, convergence, objective, time)
* `results/labels/<dataset>_<method>_<space>_seed<N>.npy` — final hard assignments
* `results/summary.csv`, `results/summary.md` — mean ± s.d. over seeds, with DMSC's no-guidance numbers (paper Table I) for comparison
