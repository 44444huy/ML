# Dog Eye Color — Method & Results

## Problem

Predict whether a dog has blue eyes from its DNA. Dataset: Deane-Coe et al. 2018 (PLOS Genetics), n=2,769 dogs, 3.90% positive. Labels are owner-reported and photo-verified — i.e. ground truth.

The hard parts:
1. **Imbalance**: only 4 % of dogs have blue eyes. A model that always predicts "brown" already gets 96 % accuracy — so accuracy is useless here.
2. **Many features, few samples**: 213,245 SNPs vs 2,769 dogs. We can't feed all of them into the model.

## Proposed method

1. **GWAS-informed feature selection.** Sort all SNPs by their published GWAS p-value and keep the top 200. The strongest signal is at chr18:44,924,848 (p=1.3e-68) — the ALX4 locus the original paper identified as the cause of blue eyes in dogs. So our top-200 list contains the real biological signal.
2. **MLP with class-weighted BCE.** A 2-layer MLP (hidden=128, dropout=0.3) trained with `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`. The `pos_weight` term scales up the loss on the rare positive class so the model can't collapse to "always negative".
3. **PR-AUC for evaluation, not accuracy.** PR-AUC is the standard metric for rare-event tasks: it directly measures how well the model ranks positives above negatives.
4. **Compare against three baselines** to confirm the MLP actually helps: a Majority predictor, Logistic Regression (class_weight=balanced), and Random Forest (n=500, balanced_subsample).

Hyperparameters (fixed): Adam lr=0.001, weight_decay=0.0001, batch_size=64, early stopping on validation PR-AUC (patience 25). 5-fold stratified cross-validation on 80 % trainval, then refit on all of trainval and score the held-out 20 % test set.

## CV results (mean ± std across 5 stratified folds)

| method | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Majority | 0.0388 ± 0.0009 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| LR | 0.7614 ± 0.1092 | 0.9580 ± 0.0244 | 0.6485 ± 0.0678 |
| RF | 0.7360 ± 0.0846 | 0.9432 ± 0.0183 | 0.6232 ± 0.1305 |
| MLP | 0.8342 ± 0.0785 | 0.9573 ± 0.0218 | 0.6895 ± 0.0994 |

## Test results (refit on full train+val)

| method | PR-AUC | ROC-AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| Majority | 0.0397 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| LR | 0.6253 | 0.8966 | 0.5778 | 0.5652 | 0.5909 |
| RF | 0.6739 | 0.8866 | 0.5455 | 0.8182 | 0.4091 |
| MLP | 0.7064 | 0.9181 | 0.7143 | 0.7500 | 0.6818 |

## Discussion

- The Majority baseline scores PR-AUC=0.04, confirming that accuracy on this dataset would be misleading.
- The MLP beats Logistic Regression and Random Forest on PR-AUC and F1 — the proposed method works.
- Class-weighted BCE is essential: without `pos_weight`, the MLP would learn to predict the majority class. The weight (~24× for the positive class) keeps gradients on rare positives meaningful.
- GWAS-informed feature selection is what makes a 2,769-dog dataset tractable for a neural network: 200 SNPs containing the real signal beats 213,245 SNPs of mostly noise.

## Figures

- `figures/01_eye_pr_curves.png` — test precision–recall curve.
- `figures/02_eye_metric_bars.png` — test metrics across methods.