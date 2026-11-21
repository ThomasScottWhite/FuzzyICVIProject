#!/bin/bash
# One-time environment setup.  Run on the LOGIN NODE from the project root:
#     bash slurm/setup_env.sh
#
# Creates a Python virtual environment in ./venv and installs requirements.txt.
# The baseline is CPU-only (NumPy / scikit-learn), so no CUDA or GPU module is needed.
set -euo pipefail

# --- Cluster-specific: load a Python >= 3.9 module if your system Python is old. -------
# Run `module avail python` to see what Mill offers, then uncomment and edit, e.g.:
# module load python/3.11

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Environment ready.  Activate with:  source venv/bin/activate"
