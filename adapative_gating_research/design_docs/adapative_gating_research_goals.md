
# Research goal

Determine whether a fixed silhouette threshold of ρ = 0.5 reliably delivers
useful clustering guidance across datasets, and whether adaptive guidance
methods improve final clustering quality when that fixed rule is ineffective.

## Experiment 1 — Characterize the fixed gate

RQ1: How does the fixed ρ = 0.5 gate behave across datasets and seeds?
Compare fixed-gate training with no guidance using matched seeds and training
settings. Record the number and timing of gate openings, adjustments actually
applied, silhouette trajectories, and final ARI and NMI.

This experiment establishes where the fixed gate is inactive, frequently
active, or sensitive to the seed. It also tests whether its guidance changes
training outcomes. An opening at the final epoch is not an applied adjustment.

## Experiment 2 — Compare adaptive guidance methods

RQ2: On a dataset where the fixed gate rarely activates, can an adaptive
method deliver guidance and improve final clustering quality over the fixed
gate and no guidance?

RQ3: On a dataset where the fixed gate does activate, do adaptive methods
retain or improve final quality?

Compare five prespecified methods—relative threshold, null-calibrated
threshold, stability gate, confidence-weighted guidance, and soft guidance—
against the fixed gate and no-guidance control. Use fresh matched seeds if
Experiment 1 informed dataset selection. Keep the training schedule, FCM
settings, evaluation points, and base adjustment strength consistent.
Prespecify each method's calibration and hyperparameters before examining
Experiment 2 outcomes.

Primary outcome: paired difference in final ARI. Secondary outcomes: final
NMI, silhouette trajectories, runtime, intervention timing, and guidance
amount. Record actual applied adjustments for binary gates and cumulative
adjustment strength for weighted or soft methods. More activation alone does
not establish improvement.

## Scope of conclusions

Experiment 1 can demonstrate limitations of the fixed threshold on the
tested datasets. Experiment 2 can show whether particular adaptive methods
improve those cases while preserving quality on an active-gate dataset.
Neither experiment establishes that one rule is best on all datasets.

A claim that intervention *timing itself* is better would require a separate
control with the same amount of guidance applied at different times.




## Old Goals
RQ1: How does the fixed threshold behave across datasets? Does it never open, opens early, or opens nearly every evaluation. How does opening adjust score tarjectories.

RQ2: How does relative thresholds change gate activation? Measure the number and timing of activations. More activations alone may not suggest sucess.

RQ3: Does it improve final clustering quality? Compare ARI and NMI across matched seeds, datasets, and training schedules.

RQ4: Does it select better intervention times? This is the most important question. Compare it with a control that applies the same number of interventions at different times. That separates timing from simply giving the model more guidance.

RQ5: Does it remain reliable across datasets and reference sizes? Test whether decisions and final quality change when silhouette is calculated on different evaluation sample sizes. A relative threshold may itself shift when the reference size changes.

RQ6: Can the relative rule open when clustering is still poor?