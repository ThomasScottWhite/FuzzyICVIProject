# Experiment 1 gamma sweep

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
quality trajectories and gate response for each dataset, plus final paired
differences across datasets. `across_gamma/` contains matched-seed gamma
responses, positive-seed counts, and gate activity. Use `--input` and `--output`
to choose other directories. Paired differences are fixed gate minus no
guidance; the quality trajectories use scores recorded at gate checks and the
following epochs.
