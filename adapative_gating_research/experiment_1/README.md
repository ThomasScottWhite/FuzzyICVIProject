# Experiment 1 gamma sweep

## HIGGS-only extension

`higgs` uses exactly 1,000,000 rows sampled without replacement from the local,
headerless `data/HIGGS.csv` (first column: binary label; remaining 28: features).
The sampling seed is fixed at 2026, independently of training seeds. Features
are min-max scaled on this subset; labels are excluded from the features.
Both experiments reuse the same cached subset and 5,000-point reference sample.

Prepare the subset once in the `fuzzyicvi` environment, then submit from the
repository root:

```bash
python -m dmsc_custom.higgs
mkdir -p adapative_gating_research/experiment_1/runs
sbatch adapative_gating_research/experiment_1/slurm_higgs.sh
```

Preparation scans the CSV twice with bounded memory and saves raw sampled rows,
zero-based source-row indices, and source metadata in
`data/cache/higgs-1000000-seed-2026.npz`. Loaders also prepare it automatically;
a file lock and atomic replacement protect concurrent Slurm tasks. A changed
source path, size, or modification time invalidates the cache. Keep the original
CSV available for source validation. `--data-home` changes the cache directory.

The separate HIGGS array contains 20 tasks: four gammas × five seeds (42–46),
each running fixed guidance and no guidance for 200 epochs. It requests 32 GB
host memory and keeps the existing GPU and time limit. Results go under
`runs/gamma_sweep_200/gamma-<value>/higgs/seed-<seed>/` and are picked up by the
existing analysis and graphs. The original Slurm script is unchanged.

The Slurm array runs the fixed 0.5 gate against no guidance for four datasets,
five paired seeds (42–46), and four adjustment strengths: 0.001, 0.01, 0.05,
and 0.1. Training remains at 200 epochs. There are 80 array tasks; each task
runs both conditions with the same dataset and seed.

Submit from the repository root after creating the Slurm log directory:

```bash
mkdir -p adapative_gating_research/experiment_1/runs
sbatch adapative_gating_research/experiment_1/slurm.sh
```

Each gamma has its own result folder:
`runs/gamma_sweep_200/gamma-<value>/<dataset>/seed-<seed>/`.
Slurm logs go directly under `runs/` as `rq1-<job>_<task>.out`.
Earlier Experiment 1 results outside `gamma_sweep_200` are left in place.

Analyze one gamma at a time, for example:

```bash
python -m adapative_gating_research.experiment_1.analyze \
  --input adapative_gating_research/experiment_1/runs/gamma_sweep_200/gamma-0.01
```

Graph the complete sweep with:

```bash
python -m adapative_gating_research.experiment_1.graphs
```

The new `graphs.py` reads completed runs under `runs/gamma_sweep_200` and
writes figures to `runs/gamma_sweep_200/graphs` by default. `by_gamma/` contains
quality trajectories and gate response for each dataset, plus a
`*_gate_response_with_fixed.png` version that overlays the orange no-guidance
silhouette trajectory on the blue fixed-gate trajectory, and final paired
differences across datasets. `across_gamma/` contains matched-seed gamma
responses, positive-seed counts, and gate activity. Use `--input` and `--output`
to choose other directories. Paired differences are fixed gate minus no
guidance; the quality trajectories use scores recorded at gate checks and the
following epochs.
