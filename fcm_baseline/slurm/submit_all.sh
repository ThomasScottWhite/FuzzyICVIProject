#!/bin/bash
# Convenience wrapper: cache data, submit the 4-task baseline array, then submit the
# aggregation job so it starts only if every array task succeeds.
#
# Run from the project root on the LOGIN NODE (it needs internet for the data step):
#     bash slurm/submit_all.sh
set -euo pipefail
mkdir -p logs results data_cache

source venv/bin/activate
echo "Caching datasets (needs internet)..."
python -m fcm_baseline.prepare_data --data-dir data_cache

ARRAY_ID=$(sbatch --parsable slurm/run_baseline.sbatch)
echo "Submitted baseline array job: ${ARRAY_ID}"

AGG_ID=$(sbatch --parsable --dependency=afterok:"${ARRAY_ID}" slurm/aggregate.sbatch)
echo "Submitted aggregation job:   ${AGG_ID} (runs after ${ARRAY_ID} succeeds)"
echo "Monitor with:  squeue -u \$USER     Logs in: logs/"
