# FuzzyICVIProject

dmsc_baseline is the frozen control implementation. Experiment-specific behavior lives in `dmsc_custom` and `experiments/`.
## Setup

Use Python 3.10 or newer. In Ubuntu/WSL, create and activate a virtual environment, then install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run the baseline control

```bash
python -m dmsc_baseline --dataset iris --seed 42 --output runs/iris-seed-42
python -m dmsc_baseline --dataset wine --seed 42 --output runs/wine-seed-42
dmsc-baseline --dataset iris --seed 42 --output runs/iris-seed-42
```