#!/bin/bash
#SBATCH --job-name=fcm_aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --partition=<your-cpu-partition>
##SBATCH --account=<your-account>
#
# Merges the per-dataset CSVs into results/summary.csv and results/summary.md.
# Normally launched by slurm/submit_all.sh with --dependency=afterok on the array job.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# module load python/3.11
source venv/bin/activate
python -m fcm_baseline.aggregate --results-dir results
