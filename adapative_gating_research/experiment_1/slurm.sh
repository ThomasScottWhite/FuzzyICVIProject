#!/bin/bash -l
#SBATCH --job-name=dmsc-rq1
#SBATCH --partition=requeue
#SBATCH --array=0-79%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:V100-SXM2-32GB:1
#SBATCH --time=1-00:00:00
#SBATCH --output=adapative_gating_research/experiment_1/runs/rq1-%A_%a.out

set -euo pipefail

# Four gamma values x four datasets x five seeds. Each task runs both conditions.
gammas=(0.001 0.01 0.05 0.1)
datasets=(iris wine covertype fashion_mnist)
seeds=(42 43 44 45 46)
task=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}
per_gamma=$((${#datasets[@]} * ${#seeds[@]}))
gamma=${gammas[$((task / per_gamma))]}
dataset=${datasets[$(((task % per_gamma) / ${#seeds[@]}))]}
seed=${seeds[$((task % ${#seeds[@]}))]}

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ ! -f adapative_gating_research/experiment_1/run.py ]]; then
    echo "Submit from the FuzzyICVIProject repository root." >&2
    exit 1
fi
module load anaconda
eval "$(conda shell.bash hook)"
conda activate fuzzyicvi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU is visible to PyTorch"; print(torch.__version__, torch.cuda.get_device_name(0), flush=True)'

python -m adapative_gating_research.experiment_1.run \
    --dataset "$dataset" \
    --seed "$seed" \
    --gamma "$gamma" \
    --output "adapative_gating_research/experiment_1/runs/gamma_sweep_200/gamma-$gamma" \
    --device cuda:0
