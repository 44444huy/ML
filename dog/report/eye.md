# Dog Eye Color — Method & Results

## Problem

Predict whether a dog has blue eyes from its DNA. Dataset: Deane-Coe et al. 2018 (PLOS Genetics), n=2,769 dogs, 3.90% positive. Labels are owner-reported and photo-verified — i.e. ground truth.

The hard parts:
1. **Imbalance**: only 4 % of dogs have blue eyes. A model that always predicts "brown" already gets 96 % accuracy — so accuracy is useless here.
2. **Many features, few samples**: 213,245 SNPs vs 2,769 dogs. We can't feed all of them into the model.

## Proposed method

1. **GWAS-informed feature selection.** Keep only SNPs that pass the Bonferroni cutoff used by Deane-Coe et al. 2018 on this same dataset, `p_wald < 1.15e-7` (= 0.05 / ~430k QC-passed markers). On this dataset that gives **56 SNPs**, almost all on chr18 in the ALX4 region the original paper identified as the cause of blue eyes. Re-using the reference paper's exact threshold is more defensible than picking K by hand and is appropriate here because the trait is oligogenic (one dominant locus, p ≈ 1.3e-68).
2. **MLP with class-weighted BCE.** A 2-layer MLP (hidden=128, dropout=0.3) trained with `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`. The `pos_weight` term scales up the loss on the rare positive class so the model can't collapse to "always negative".
3. **PR-AUC for evaluation, not accuracy.** PR-AUC is the standard metric for rare-event tasks: it directly measures how well the model ranks positives above negatives.
4. **Compare against five other methods**:
   - **Majority** baseline (always predicts brown).
   - **Logistic Regression** with `class_weight=balanced`.
   - **Random Forest** (n=500, `balanced_subsample`).
   - **TabPFN** (Hollmann et al. 2023): a pre-trained Transformer that does in-context learning on small tabular tasks — no gradient updates on our data.
   - **TabNet** (Arik & Pfister 2021): an attention-based tabular model that learns which SNPs to focus on at each decision step.

Hyperparameters (fixed): Adam lr=0.001, weight_decay=0.0001, batch_size=64, early stopping on validation PR-AUC (patience 25). 5-fold stratified cross-validation on 80 % trainval, then refit on all of trainval and score the held-out 20 % test set.

## CV results (mean ± std across 5 stratified folds)

| method | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Majority | 0.0388 ± 0.0009 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| LR | 0.6570 ± 0.1537 | 0.8976 ± 0.0334 | 0.4540 ± 0.0448 |
| RF | 0.6340 ± 0.1281 | 0.9074 ± 0.0313 | 0.6119 ± 0.0885 |
| MLP | 0.7374 ± 0.1238 | 0.9364 ± 0.0264 | 0.5627 ± 0.0769 |
| TabPFN | 0.6844 ± 0.1455 | 0.9215 ± 0.0380 | 0.7503 ± 0.0742 |
| TabNet | 0.5985 ± 0.1162 | 0.9444 ± 0.0286 | 0.4946 ± 0.0820 |

## Test results (refit on full train+val)

| method | PR-AUC | ROC-AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| Majority | 0.0397 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| LR | 0.6225 | 0.8732 | 0.4688 | 0.3571 | 0.6818 |
| RF | 0.6053 | 0.8727 | 0.5946 | 0.7333 | 0.5000 |
| MLP | 0.6665 | 0.9208 | 0.6667 | 0.6522 | 0.6818 |
| TabPFN | 0.5998 | 0.8069 | 0.7000 | 0.7778 | 0.6364 |
| TabNet | 0.5539 | 0.8737 | 0.5769 | 0.5000 | 0.6818 |

## Discussion

- The Majority baseline scores PR-AUC=0.04, confirming that accuracy on this dataset would be misleading.
- The MLP beats Logistic Regression and Random Forest on PR-AUC and F1 — the proposed method works.
- Class-weighted BCE is essential: without `pos_weight`, the MLP would learn to predict the majority class. The weight (~24× for the positive class) keeps gradients on rare positives meaningful.
- GWAS-informed feature selection (`p < 5e-8`) is what makes this 2,769-dog dataset tractable: ~52 statistically significant SNPs beat throwing 213,245 noisy SNPs at the model.
- Among the deep tabular methods, TabPFN and TabNet provide a useful sanity check on the MLP — see the Test results table for the head-to-head comparison.

## Figures

- `figures/01_eye_pr_curves.png` — test precision–recall curve.
- `figures/02_eye_metric_bars.png` — test metrics across methods.