# Shared Conda setup on Missouri S&T's Mill

Create one environment named `fuzzyicvi` for this project

```bash
cd /home/thomas/FuzzyICVIProject
module load anaconda
eval "$(conda shell.bash hook)"
conda create -n fuzzyicvi python=3.11 pip -y
conda activate fuzzyicvi

# CUDA 12.6 PyTorch wheels retain support for Mill's V100 (Volta) GPUs.
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

```bash
python -c "import torch, sklearn; import dmsc_custom; print(torch.__version__, sklearn.__version__)"
```