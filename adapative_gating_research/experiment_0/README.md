# Experiment 0: silhouette timing

Run from the repository root:

```bash
python -m adapative_gating_research.experiment_0.run --dataset iris
```

For a larger benchmark, use `--dataset covertype --reference-size 2000 --epochs 20` after the dataset is cached. Other options include `--evaluation-interval`, `--seed`, `--device`, and `--output`.

`summary.json` compares cumulative exact silhouette time with the rest of `fit()` wall time. It also compares silhouette with the other measured stages: autoencoder training, full embedding encoding, and FCM. Fit wall time includes setup and Python overhead; data loading is reported separately. `evaluations.csv` records each scheduled silhouette call; if the final epoch is not scheduled, the trainer also scores once at the end and that call is included in the summary only. Timings are wall clock measurements, so repeat runs under similar machine load for a stable comparison. The silhouette uses at most `--reference-size` points, while training and FCM use all samples.

On Mill, submit `sbatch adapative_gating_research/experiment_0/slurm.sh` from the repository root after ensuring `adapative_gating_research/experiment_0/runs` exists. The array runs seeds 42–44 on Iris, Wine, Covertype, and Fashion-MNIST (12 tasks total, at most two concurrently). Each timing run uses 20 epochs, checks at epochs 10 and 20, and a fixed silhouette subset capped at 5,000 points. Results are saved under `runs/bench_5000_e20_i10/<dataset>/`; Slurm logs go directly under `runs/`.
