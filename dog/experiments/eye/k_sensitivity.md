# K sensitivity (MLP)

All numbers below come from the same fixed splits (seed=42). CV = 5-fold stratified CV-mean on trainval. TEST = refit on full trainval, evaluated on the held-out 20 % test set.

| Config | #SNPs | CV PR-AUC | CV ROC-AUC | CV F1 | TEST PR-AUC | TEST ROC-AUC | TEST F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p<1.15e-7 (Deane-Coe Bonferroni, default) | 56 | 0.737 | 0.936 | 0.563 | 0.667 | 0.921 | 0.667 |
| p<5e-8 (genome-wide standard) | 52 | 0.727 | 0.921 | 0.550 | 0.641 | 0.868 | 0.536 |
| p<1e-5 (suggestive) | 96 | 0.765 | 0.925 | 0.611 | 0.624 | 0.822 | 0.667 |
| p<1e-4 (lenient) | 157 | 0.797 | 0.934 | 0.679 | 0.674 | 0.856 | 0.632 |
| top_k=100 | 100 | 0.751 | 0.924 | 0.593 | 0.609 | 0.840 | 0.595 |
| top_k=200 | 200 | 0.834 | 0.957 | 0.689 | 0.706 | 0.918 | 0.714 |
| top_k=500 | 500 | 0.915 | 0.991 | 0.817 | 0.836 | 0.958 | 0.732 |

Default in main pipeline: **`p < 1.15e-7`** (Bonferroni cutoff used by Deane-Coe et al. 2018 on this dataset: 0.05 / ~430k QC-passed markers).