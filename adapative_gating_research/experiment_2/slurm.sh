#!/bin/bash -l
#SBATCH --job-name=dmsc-rq2
#SBATCH --partition=requeue
#SBATCH --array=0-69%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=adapative_gating_research/experiment_2/runs/rq2-%A_%a.out

set -euo pipefail

# Two datasets x seven conditions x five fresh matched seeds.
datasets=(covertype iris)
conditions=(no_guidance fixed relative null_calibrated stability confidence_weighted soft)
seeds=(101 102 103 104 105)
task=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}
per_dataset=$((${#conditions[@]} * ${#seeds[@]}))
dataset=${datasets[$((task / per_dataset))]}
condition=${conditions[$(((task / ${#seeds[@]}) % ${#conditions[@]}))]}
seed=${seeds[$((task % ${#seeds[@]}))]}

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ ! -f adapative_gating_research/experiment_2/run.py ]]; then
    echo "Submit from the FuzzyICVIProject repository root." >&2
    exit 1
fi
module load anaconda
eval "$(conda shell.bash hook)"
conda activate fuzzyicvi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU is visible to PyTorch"; print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'

python -m adapative_gating_research.experiment_2.run \
    --dataset "$dataset" \
    --condition "$condition" \
    --seed "$seed" \
    --epochs 200 \
    --evaluation-interval 10 \
    --reference-size 5000 \
    --gamma 0.01 \
    --output adapative_gating_research/experiment_2/runs/paired_v1 \
    --device cuda:0
