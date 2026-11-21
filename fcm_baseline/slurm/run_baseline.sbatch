#!/bin/bash
#SBATCH --job-name=fcm_baseline
#SBATCH --array=0-3                    # one array task per dataset (see DATASETS below)
#SBATCH --cpus-per-task=4              # matches the 4 cores used for the DMSC runs
#SBATCH --mem=16G                      # Covertype peaks well under this
#SBATCH --time=04:00:00                # generous; Iris/Wine finish in seconds
#SBATCH --output=logs/%x_%A_%a.out     # %A = array job id, %a = array task index
#SBATCH --error=logs/%x_%A_%a.err
##SBATCH --partition=<your-cpu-partition>   # <-- uncomment/edit if your site requires one
##SBATCH --account=<your-account>           # <-- uncomment/edit if your site requires one
#
# Runs the independent FCM / K-means baselines.  Each array task handles ONE dataset, so
# the four datasets run in parallel as separate jobs.
#
# Submit from the project root:   sbatch slurm/run_baseline.sbatch
# (or use slurm/submit_all.sh, which also chains the aggregation step).
#
# Safe to resubmit: finished (method, space, seed) runs are skipped automatically.

set -euo pipefail

DATASETS=(iris wine covertype fashion_mnist)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Always run from the project root regardless of where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

# --- Environment -----------------------------------------------------------------------
# module load python/3.11            # <-- same module you used in setup_env.sh, if any
source venv/bin/activate

# Keep BLAS / OpenMP thread counts equal to the cores we were allocated.  Without this,
# NumPy may spawn one thread per *node* core and be throttled by Slurm's cgroup.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "Job ${SLURM_JOB_ID} | array task ${SLURM_ARRAY_TASK_ID} | dataset ${DATASET} | node $(hostname)"
date

# Compute nodes may be offline: the data cache must have been filled beforehand with
#   python -m fcm_baseline.prepare_data --data-dir data_cache      (on the login node)
python -m fcm_baseline.run_baseline \
    --datasets "${DATASET}" \
    --methods fcm kmeans \
    --spaces raw pca3 \
    --seeds 42 43 44 45 46 \
    --data-dir data_cache \
    --output-dir results

date
echo "Done: ${DATASET}"
