How does the fixed threshold behave across datasets? Does it never open, opens early, or opens nearly every evaluation. How does opening adjust score tarjectories.

How does relative thresholds change gate activation? Measure the number and timing of activations. More activations alone may not suggest sucess.

Does it improve final clustering quality? Compare ARI and NMI across matched seeds, datasets, and training schedules.

Does it select better intervention times? This is the most important question. Compare it with a control that applies the same number of interventions at different times. That separates timing from simply giving the model more guidance.

Does it remain reliable across datasets and reference sizes? Test whether decisions and final quality change when silhouette is calculated on different evaluation sample sizes. A relative threshold may itself shift when the reference size changes.

Can the relative rule open when clustering is still poor?