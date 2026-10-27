#!/bin/bash -l
#SBATCH --job-name=dmsc-e0
#SBATCH --partition=requeue
#SBATCH --array=0-11%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=adapative_gating_research/experiment_0/runs/e0-%A_%a.out

set -euo pipefail

# Four datasets x three independent timing runs, one run per array task.
datasets=(iris wine covertype fashion_mnist)
seeds=(42 43 44)
task=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}
dataset=${datasets[$((task / ${#seeds[@]}))]}
seed=${seeds[$((task % ${#seeds[@]}))]}

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ ! -f adapative_gating_research/experiment_0/run.py ]]; then
    echo "Submit from the FuzzyICVIProject repository root." >&2
    exit 1
fi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate fuzzyicvi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU is visible to PyTorch"; print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'

python -m adapative_gating_research.experiment_0.run \
    --dataset "$dataset" \
    --seed "$seed" \
    --epochs 20 \
    --evaluation-interval 10 \
    --reference-size 5000 \
    --output adapative_gating_research/experiment_0/runs/bench_5000_e20_i10 \
    --device cuda:0
