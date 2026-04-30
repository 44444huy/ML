# Baseline Results

Three classifiers (Logistic Regression, SVM-RBF, Random Forest) trained per trait with balanced class weights, 5-fold stratified CV on the train+val split, then refit and scored on the held-out 20% test.


## CV validation (mean ± std across 5 folds)

| model | trait | accuracy | macro_f1 | auc_ovr | ece |
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

## Test metrics (refit on full train+val)

| model | trait | accuracy | macro_f1 | auc_ovr | ece | mae | qwk |
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

## Key takeaways

1. **Accuracy is misleading.** All three models reach 92–100% accuracy on every trait, yet macro-F1 sits at 0.42–0.67 because rare classes (hair `red`, skin `pale`, eye `intermediate`) are collapsed by argmax and therefore never predicted.
2. **Hair and Skin are far from solved.** Even the best model reaches only macro-F1 ≈ 0.55; ordinal information (intended order of classes) is being ignored because CE loss treats them as categorical.
3. **Eye collapses to 2 classes.** The `intermediate` class never wins argmax in HIrisPlex output, so it is absent from train/test. Any classifier trained on these labels cannot predict it.
4. **Baselines essentially replicate HIrisPlex.** They map 41 SNPs to the same kind of output HIrisPlex produced — by construction they cannot exceed the silver standard.

Confusion matrices: `figures/07_baseline_confusion_matrices.png`.
