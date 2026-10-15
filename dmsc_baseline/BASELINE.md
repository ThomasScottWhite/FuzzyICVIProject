# Baseline control snapshot

This package is the control implementation for the project. It was separated from the original `dmsc` package before Experiment 1 or Experiment 3 changes were made. The custom package began as an independent copy of this snapshot.

Keep experiment-specific behavior in `dmsc_custom`. Do not change this package to add schedules, alternate validation measures, or stopping rules. If a reproduction defect is discovered, record the issue and the correction explicitly so the control remains interpretable.

The package implements the method described in the paper, with documented choices for details the paper leaves unspecified. In particular, this snapshot uses Adam, mean squared reconstruction error, batch size 256, and FCM tolerance `1e-4`. These choices are not claimed to reproduce the authors' software bit-for-bit; the paper does not provide their implementation.

Run the control with `python -m dmsc_baseline` or the `dmsc-baseline` command. Its default outputs use the `dmsc-baseline-` prefix to keep them separate from custom runs.
