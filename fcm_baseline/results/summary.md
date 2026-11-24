# Independent baseline results (mean ± s.d. over seeds)

| Dataset | Method | Space | Seeds | ACC | NMI | ARI |
|---|---|---|---|---|---|---|
| covertype | fcm | pca3 | 5 | 0.285 ± 0.011 | 0.139 ± 0.009 | 0.050 ± 0.005 |
| covertype | fcm | raw | 5 | 0.360 ± 0.004 | 0.104 ± 0.003 | 0.036 ± 0.003 |
| covertype | kmeans | pca3 | 5 | 0.293 ± 0.011 | 0.144 ± 0.002 | 0.052 ± 0.003 |
| covertype | kmeans | raw | 5 | 0.285 ± 0.013 | 0.142 ± 0.003 | 0.038 ± 0.006 |
| fashion_mnist | fcm | pca3 | 5 | 0.461 ± 0.017 | 0.450 ± 0.003 | 0.295 ± 0.011 |
| fashion_mnist | fcm | raw | 5 | 0.267 ± 0.001 | 0.250 ± 0.002 | 0.171 ± 0.001 |
| fashion_mnist | kmeans | pca3 | 5 | 0.467 ± 0.001 | 0.456 ± 0.001 | 0.298 ± 0.001 |
| fashion_mnist | kmeans | raw | 5 | 0.509 ± 0.047 | 0.518 ± 0.013 | 0.365 ± 0.024 |
| iris | fcm | pca3 | 5 | 0.880 ± 0.000 | 0.723 ± 0.000 | 0.702 ± 0.000 |
| iris | fcm | raw | 5 | 0.893 ± 0.000 | 0.743 ± 0.000 | 0.729 ± 0.000 |
| iris | kmeans | pca3 | 5 | 0.887 ± 0.000 | 0.742 ± 0.000 | 0.716 ± 0.000 |
| iris | kmeans | raw | 5 | 0.887 ± 0.000 | 0.742 ± 0.000 | 0.716 ± 0.000 |
| wine | fcm | pca3 | 5 | 0.955 ± 0.000 | 0.847 ± 0.000 | 0.865 ± 0.000 |
| wine | fcm | raw | 5 | 0.949 ± 0.000 | 0.834 ± 0.000 | 0.850 ± 0.000 |
| wine | kmeans | pca3 | 5 | 0.949 ± 0.000 | 0.835 ± 0.000 | 0.847 ± 0.000 |
| wine | kmeans | raw | 5 | 0.951 ± 0.003 | 0.844 ± 0.005 | 0.857 ± 0.007 |

## DMSC no-guidance reference (paper Table I, epoch 200)

| Dataset | ACC | NMI | ARI |
|---|---|---|---|
| iris | 0.905 ± 0.039 | 0.769 ± 0.069 | 0.759 ± 0.084 |
| wine | 0.925 ± 0.015 | 0.763 ± 0.038 | 0.780 ± 0.041 |
| covertype | 0.310 ± 0.029 | 0.128 ± 0.029 | 0.062 ± 0.016 |
| fashion_mnist | 0.499 ± 0.029 | 0.509 ± 0.027 | 0.337 ± 0.025 |

_Silhouette values in `summary.csv` are computed in the clustered space (raw or PCA-3) and are not comparable to DMSC's latent-space silhouette._
