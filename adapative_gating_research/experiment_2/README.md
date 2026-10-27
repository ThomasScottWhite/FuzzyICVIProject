# Experiment 2: compare adaptive guidance methods

This experiment compares seven conditions on the same autoencoder, FCM,
evaluation schedule, reference sample, and paired seeds. Each gamma value is
shared across the guided conditions. The conditions are no guidance, the fixed
silhouette gate at 0.5, and five adaptive approaches. Each check at epoch 10,
20, etc. controls training
from the following epoch through the next check. A decision at the final epoch
does not count as applied guidance.

The gamma sweep uses `0.001`, `0.01`, `0.05`, and `0.1` for each of the six
guided conditions. No guidance is run once per dataset and seed, with gamma 0.
With two datasets and five seeds, this is 250 training runs. Every run trains
for 1,000 epochs; `summary.json` records results at both epochs 200 and 1,000.
The first 200 epochs can therefore be analyzed without rerunning training.
The scheduled datasets are Covertype and Iris, with seeds 101–105.

From the repository root on the Mill, create the Slurm log directory before
submitting:

```bash
mkdir -p adapative_gating_research/experiment_2/runs
sbatch adapative_gating_research/experiment_2/slurm.sh
```

Results are saved under
`runs/gamma_sweep_1000/<dataset>/<condition>/gamma-<value>/seed-<seed>/`.
Run `python -m adapative_gating_research.experiment_2.analyze` from the
repository root after the jobs finish. The `analysis` directory contains
paired per-seed results, aggregate results, missing-run diagnostics, and
gamma-response plots for epochs 200 and 1,000. The requeue partition can
preempt a run; the current trainer starts that run again from epoch 1.

| Condition | Prespecified rule at each check |
| --- | --- |
| `no_guidance` | Never adjust. |
| `fixed` | Adjust when the reference silhouette is at least 0.5. |
| `relative` | First collect five valid silhouette scores without guidance. Freeze the 75th percentile, with a safety floor of 0.0; subsequent scores at or above it activate guidance. |
| `null_calibrated` | Randomly permute the reference FCM labels 19 times, preserving cluster sizes. Activate when the observed silhouette exceeds all 19 null scores (empirical one-sided p ≤ 0.05). |
| `stability` | Add independent Gaussian noise at 2% of each latent coordinate's standard deviation twice, rerun full-data FCM, and activate when mean assignment ARI on the reference sample is at least 0.9. This ARI compares unsupervised partitions and does not use ground-truth labels. |
| `confidence_weighted` | Adjust every point with weight `max(0, local centroid silhouette) × normalized maximum FCM membership`, both in [0, 1]. This uses a centroid approximation to local silhouette for tractability on large datasets. |
| `soft` | Use strength `sigmoid((reference silhouette − 0.5) / 0.05)` instead of a binary gate. |
