#!/bin/bash -l
#SBATCH --job-name=dmsc-rq1
#SBATCH --partition=requeue
#SBATCH --array=0-19%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=adapative_gating_research/experiment_1/runs/rq1-%A_%a.out

set -euo pipefail

# Each array task runs both conditions for one dataset and one seed.
datasets=(iris wine covertype fashion_mnist)
seeds=(42 43 44 45 46)
task=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}
dataset=${datasets[$((task / ${#seeds[@]}))]}
seed=${seeds[$((task % ${#seeds[@]}))]}

cd "$SLURM_SUBMIT_DIR"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate fuzzyicvi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU is visible to PyTorch"; print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'

python -m adapative_gating_research.experiment_1.run \
    --dataset "$dataset" \
    --seed "$seed" \
    --device cuda:0
