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

Analyze or graph one gamma at a time, for example:

```bash
python -m adapative_gating_research.experiment_1.analyze \
  --input adapative_gating_research/experiment_1/runs/gamma_sweep_200/gamma-0.01
python -m adapative_gating_research.experiment_1.graph \
  --input adapative_gating_research/experiment_1/runs/gamma_sweep_200/gamma-0.01
```
