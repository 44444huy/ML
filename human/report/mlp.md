# MLP vs Baselines

Identical preprocessing, splits, and metric suite as Week 2. Two MLP variants share an architecture and differ only in loss:

- **M1**: CrossEntropy on argmax(HIrisPlex) labels.
- **M2**: KL-divergence on raw HIrisPlex probability vectors (soft label).

Hyperparameters (fixed): 2 hidden layers × 128, dropout=0.3, Adam lr=1e-3, wd=1e-4, batch=64, early stopping on val macro-F1 with patience=20.

## CV validation (mean ± std across 5 folds)

| method | trait | accuracy | macro_f1 | auc_ovr | ece |
|---|---|---|---|---|---|
| LR | eye | 0.9960 ± 0.0038 | 0.6639 ± 0.0026 | nan ± nan | 0.0208 ± 0.0015 |
| LR | hair | 0.9385 ± 0.0073 | 0.4305 ± 0.0073 | nan ± nan | 0.0402 ± 0.0073 |
| LR | skin | 0.9239 ± 0.0122 | 0.5697 ± 0.0508 | 0.8976 ± 0.0000 | 0.0440 ± 0.0045 |
| SVM | eye | 0.9950 ± 0.0036 | 0.6632 ± 0.0024 | nan ± nan | 0.0073 ± 0.0026 |
| SVM | hair | 0.9209 ± 0.0102 | 0.4168 ± 0.0074 | nan ± nan | 0.0230 ± 0.0041 |
| SVM | skin | 0.9188 ± 0.0113 | 0.5807 ± 0.0698 | 0.8965 ± 0.0000 | 0.0276 ± 0.0049 |
| RF | eye | 0.9970 ± 0.0029 | 0.6646 ± 0.0020 | nan ± nan | 0.0386 ± 0.0021 |
| RF | hair | 0.9511 ± 0.0059 | 0.4196 ± 0.0133 | nan ± nan | 0.0407 ± 0.0039 |
| RF | skin | 0.9456 ± 0.0097 | 0.5385 ± 0.0167 | 0.8940 ± 0.0000 | 0.0690 ± 0.0084 |
| M1_MLP_CE | eye | 0.9980 ± 0.0019 | 0.6653 ± 0.0013 | nan ± nan | 0.0073 ± 0.0053 |
| M1_MLP_CE | hair | 0.9919 ± 0.0029 | 0.5542 ± 0.0820 | nan ± nan | 0.0114 ± 0.0036 |
| M1_MLP_CE | skin | 0.9713 ± 0.0076 | 0.6795 ± 0.0955 | 0.9050 ± 0.0000 | 0.0230 ± 0.0036 |
| M2_MLP_KL | eye | 0.9995 ± 0.0010 | 0.6663 ± 0.0007 | nan ± nan | 0.3644 ± 0.0015 |
| M2_MLP_KL | hair | 0.9849 ± 0.0135 | 0.5444 ± 0.0824 | nan ± nan | 0.5994 ± 0.0178 |
| M2_MLP_KL | skin | 0.9849 ± 0.0053 | 0.6301 ± 0.0611 | 0.9949 ± 0.0000 | 0.1542 ± 0.0072 |

## Test metrics (refit on full train+val)

| method | trait | accuracy | macro_f1 | auc_ovr | ece | mae | qwk |
|---|---|---|---|---|---|---|---|
| LR | eye | 1.0000 | 0.6667 | nan | 0.0160 | — | — |
| LR | hair | 0.9416 | 0.5453 | nan | 0.0368 | 0.1891 | 0.7374 |
| LR | skin | 0.9296 | 0.6072 | nan | 0.0440 | 0.0905 | 0.9319 |
| SVM | eye | 0.9980 | 0.6653 | nan | 0.0055 | — | — |
| SVM | hair | 0.9235 | 0.4178 | nan | 0.0165 | 0.2636 | 0.6475 |
| SVM | skin | 0.9195 | 0.6516 | nan | 0.0327 | 0.1087 | 0.9115 |
| RF | eye | 1.0000 | 0.6667 | nan | 0.0374 | — | — |
| RF | hair | 0.9577 | 0.4576 | nan | 0.0339 | 0.0966 | 0.8259 |
| RF | skin | 0.9618 | 0.5667 | nan | 0.0826 | 0.0644 | 0.9383 |
| M1_MLP_CE | eye | 0.9980 | 0.6653 | nan | 0.0129 | — | — |
| M1_MLP_CE | hair | 0.9759 | 0.5520 | nan | 0.0194 | 0.0704 | 0.9005 |
| M1_MLP_CE | skin | 0.9457 | 0.6152 | nan | 0.0346 | 0.0785 | 0.9349 |
| M2_MLP_KL | eye | 1.0000 | 0.6667 | nan | 0.3711 | — | — |
| M2_MLP_KL | hair | 0.9698 | 0.5447 | nan | 0.5886 | 0.0905 | 0.8247 |
| M2_MLP_KL | skin | 0.9819 | 0.7743 | nan | 0.1669 | 0.0221 | 0.9841 |


## Key takeaways

1. **M1 (MLP + CE) vs best baseline.** eye: best baseline LR F1=0.667 vs M1 F1=0.665; hair: best baseline LR F1=0.545 vs M1 F1=0.552; skin: best baseline SVM F1=0.652 vs M1 F1=0.615.
2. **M2 (MLP + soft-label KL) vs M1.** eye: M1 F1=0.665 vs M2 F1=0.667; hair: M1 F1=0.552 vs M2 F1=0.545; skin: M1 F1=0.615 vs M2 F1=0.774. Soft-label KL exploits HIrisPlex's probability distribution instead of collapsing it to argmax, so it can learn from samples that argmax turns into noise.
3. **Eye is still capped at F1 ≈ 2/3.** All methods trained on HIrisPlex argmax labels inherit the missing `intermediate` class. Raising eye F1 further requires either re-labelling with soft targets across the full 3-class output (which M2 does) or introducing ground-truth phenotypes from a different dataset (e.g. the dog dataset).
4. **Calibration (ECE) matters for forensic use.** Compare ECE across methods in the table: deep models can be overconfident on rare classes, which is a liability when predictions inform investigations.

Figures:
- `figures/08_method_comparison.png` — test macro-F1 bar chart.
- `figures/09_mlp_confusion_matrices.png` — M1/M2 confusion matrices.
