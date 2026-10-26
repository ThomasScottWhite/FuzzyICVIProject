### Relative threshold

**Decision or adjustment rule:** Let `s_t` be the mean silhouette at evaluation `t`. From the first `m` valid scores, set `theta = percentile(S_early, p)`. At later evaluations, trigger when `s_t > theta`.

**Calibration:** Use the same fixed evaluation subset for the first `m` valid scheduled evaluations. Freeze `theta` after calibration; do not update it during training or include the current score before making a decision.

**Training behavior:** The first eligible evaluation is the one after calibration. Keep the gate closed if calibration cannot obtain `m` valid scores or the current score is undefined. A trigger schedules centroid adjustment for the next epoch only; a final-epoch trigger has no training effect.

**Parameters:** Test `p = 75` and `90`, initially with `m = 5`. Fix the evaluation schedule, evaluation subset, comparison operator (`>`), and centroid-adjustment coefficient before running experiments.

**Cost:** No extra FCM runs or silhouette calculations beyond the existing scheduled evaluations. Store `m` calibration scores and one threshold.

**Failure mode:** A poor run still has scores above its own 75th percentile, so the gate may trust an incorrect partition. A frozen early threshold may also become too easy to exceed later.

**Recorded outputs:** Current score, calibration scores, percentile, frozen threshold, gate decision, trigger epoch, applied intervention epoch, and adjustment magnitude.

### Null-calibrated threshold

**Decision or adjustment rule:** At evaluation `t`, randomly permute the FCM hard labels `B` times while preserving cluster sizes. Compute a silhouette `s_t^(b)` for each permutation and `p_t = (1 + count_b[s_t^(b) >= s_t]) / (B + 1)`. Trigger when `p_t <= alpha`.

**Calibration:** The null distribution is generated from the current latent points and cluster sizes at each scheduled evaluation, then discarded. It is recalculated as training changes the embedding; it is not a frozen early-run distribution. Bootstrap FCM reruns are a stability test, not this randomized-label null.

**Training behavior:** The first scheduled evaluation is eligible. Keep the gate closed if the observed silhouette or required null scores are undefined. A trigger schedules centroid adjustment for the next epoch only; a final-epoch trigger has no training effect.

**Parameters:** Initially test `B = 199` and `alpha = 0.05`. Fix the label-permutation procedure, random-seed scheme, reference size, comparison operator (`>=`), and handling of degenerate partitions before experiments.

**Cost:** Up to `B` extra silhouette calculations per evaluation, with corresponding runtime and temporary memory costs; no extra FCM runs.

**Failure mode:** Random labels may be an easy comparison, so an incorrect but geometrically separated partition can pass. Repeated testing over epochs means `alpha` is not a run-level false-activation guarantee.

**Recorded outputs:** Observed score, null-score distribution, `p_t`, `alpha`, gate decision, trigger epoch, applied intervention epoch, adjustment magnitude, and validation time.

### Stability gate

**Decision or adjustment rule:** Perturb the latent representations `R` times, rerun FCM for each perturbation, and compare each result with the unperturbed partition on the same points using adjusted Rand index (ARI). Trigger when the median ARI is at least `a_min`.

**Calibration:** No early-run distribution is needed. Set the perturbation scale from a prespecified rule applied to the current latent representation at each evaluation. Recalculate stability at every scheduled evaluation; it is not carried forward as a threshold.

**Training behavior:** The first scheduled evaluation is eligible. Keep the gate closed if the reference partition is invalid or fewer than the prespecified minimum number of reruns succeeds. A trigger schedules centroid adjustment for the next epoch only; a final-epoch trigger has no training effect.

**Parameters:** Initially test `R = 5` and `a_min = 0.9`, with a small prespecified range of perturbation scales. Fix the perturbation type, FCM settings, random-seed scheme, minimum valid reruns, and agreement metric before experiments.

**Cost:** `R` extra FCM runs and ARI calculations per evaluation, plus storage for the perturbed assignments. No extra silhouette calculations are required for the decision.

**Failure mode:** FCM may consistently produce the same incorrect split. Very small perturbations can make poor partitions appear stable; large perturbations can reject useful ones.

**Recorded outputs:** Current silhouette as a diagnostic, perturbation scale, individual and median ARI scores, stability threshold, gate decision, trigger epoch, applied intervention epoch, adjustment magnitude, and extra runtime.

### Confidence-weighted adjustment

**Decision or adjustment rule:** For point `i`, let `s_i` be its local silhouette and `u_(1), u_(2)` its two largest FCM memberships. Set `w_i = clip(max(0, s_i) * (u_(1) - u_(2)), 0, 1)` and apply `z'_i = z_i + gamma_max * w_i * (c_{y_i} - z_i)`.

**Calibration:** No early-run distribution or global threshold is used. Compute point weights from the current FCM partition at every scheduled evaluation; do not carry them over to later evaluations.

**Training behavior:** The first scheduled evaluation is eligible. Give zero weight to a point with undefined local silhouette or membership confidence. Use the weights for adjustment in the next epoch only; weights computed at the final epoch have no training effect.

**Parameters:** Initially use `gamma_max = 0.001`. Fix the local-silhouette definition, treatment of singleton clusters and tied memberships, reference population, and weight formula before experiments. A centroid-distance proxy must be identified as a separate approximation.

**Cost:** Local silhouettes are needed for every point receiving adjustment. Exact full-data calculation may be much more expensive than the existing mean silhouette on a reference subset and requires storing one weight per training point; no extra FCM run is required.

**Failure mode:** A confidently wrong FCM assignment can still receive a large weight. The weighting formula may also reduce total guidance so much that a quality difference reflects adjustment strength rather than better point selection.

**Recorded outputs:** Mean silhouette, local silhouettes or their distribution, membership margins, point weights, fraction with zero weight, effective adjustment decision, applied intervention epoch, and total and mean adjustment magnitude.

### Soft gate

**Decision or adjustment rule:** From early scores set `L = percentile(S_early, 25)` and `H = percentile(S_early, 90)`. At later evaluations use `w_t = clip((s_t - L) / max(H - L, epsilon), 0, 1)` and `gamma_t = gamma_max * w_t`; apply `z'_i = z_i + gamma_t(c_{y_i} - z_i)`.

**Calibration:** Use the same fixed evaluation subset for the first `m` valid scheduled evaluations. Freeze `L` and `H` after calibration. Online recalibration would be a separate method because guidance changes later scores.

**Training behavior:** The first eligible evaluation is the one after calibration. Set `w_t = 0` if the current score is undefined, calibration is incomplete, or `H <= L`. Apply nonzero guidance in the next epoch only; a final-epoch weight has no training effect.

**Parameters:** Initially test `m = 5` and `gamma_max = 0.001`; fix `epsilon`, percentile interpolation, evaluation subset, clipping rule, and the `H <= L` fallback before experiments.

**Cost:** No extra FCM runs or silhouette calculations beyond existing evaluations. Store `m` calibration scores and the current global weight.

**Failure mode:** If `H - L` is small, minor score changes can create large strength changes. A poor early distribution can still assign full strength to a poor later partition.

**Recorded outputs:** Current score, calibration scores, `L`, `H`, `w_t`, `gamma_t`, effective adjustment decision, applied intervention epoch, and cumulative adjustment magnitude.
