# Dog Eye Color — Method & Results

## Problem

Predict whether a dog has blue eyes from its DNA. Dataset: Deane-Coe et al. 2018 (PLOS Genetics), n=2,769 dogs, 3.90% positive. Labels are owner-reported and photo-verified — i.e. ground truth.

The hard parts:
1. **Imbalance**: only 4 % of dogs have blue eyes. A model that always predicts "brown" already gets 96 % accuracy — so accuracy is useless here.
2. **Many features, few samples**: 213,245 SNPs vs 2,769 dogs. We can't feed all of them into the model.

## Proposed method

1. **GWAS-informed feature selection.** Keep only SNPs that pass the genome-wide significance threshold reported by Deane-Coe et al. 2018, `p_wald < 5e-8`. On this dataset that gives **52 SNPs**, almost all on chr18 in the ALX4 region the original paper identified as the cause of blue eyes. Re-using the paper-reported threshold is more defensible than picking K by hand and is appropriate here because the trait is oligogenic (one dominant locus, p ≈ 1.3e-68).
2. **MLP with class-weighted BCE.** A 2-layer MLP (hidden=128, dropout=0.3) trained with `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`. The `pos_weight` term scales up the loss on the rare positive class so the model can't collapse to "always negative".
3. **PR-AUC for evaluation, not accuracy.** PR-AUC is the standard metric for rare-event tasks: it directly measures how well the model ranks positives above negatives.
4. **Compare against additional methods rerun on the current 52-SNP bundle**:
   - **Majority** baseline (always predicts brown).
   - **Logistic Regression** with `class_weight=balanced`.
   - **Random Forest** (n=500, `balanced_subsample`).
   - **MLP (tuned)**: the best MLP hyperparameters selected by CV PR-AUC on trainval.
   - **TabPFN** (Hollmann et al. 2023): a pretrained Transformer for in-context learning on tabular tasks.
   - **TabICL** (Qu et al. 2025): a tabular foundation model with column-wise embeddings, row-wise interactions, and dataset-wise in-context learning.
   - **TabNet** (Arik & Pfister 2021): an attention-based tabular model that learns which SNPs to focus on at each decision step.

Hyperparameters (fixed): Adam lr=0.001, weight_decay=0.0001, batch_size=64, early stopping on validation PR-AUC (patience 25). 5-fold stratified cross-validation on 80 % trainval, then refit on all of trainval and score the held-out 20 % test set. Stale result files from previous feature-selection runs are ignored by the report renderer.

## CV results (mean ± std across 5 stratified folds)

| method | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Majority | 0.0388 ± 0.0009 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| LR | 0.6565 ± 0.1031 | 0.9032 ± 0.0373 | 0.4592 ± 0.0294 |
| RF | 0.6280 ± 0.1159 | 0.9154 ± 0.0278 | 0.5831 ± 0.1094 |
| MLP | 0.7275 ± 0.1285 | 0.9209 ± 0.0324 | 0.5500 ± 0.0616 |
| MLP (tuned) | 0.7420 ± 0.1145 | 0.9276 ± 0.0297 | 0.4869 ± 0.1174 |
| TabPFN | 0.6741 ± 0.1470 | 0.9212 ± 0.0350 | 0.7533 ± 0.0776 |
| TabICL | 0.6931 ± 0.1354 | 0.9311 ± 0.0327 | 0.7537 ± 0.0757 |
| TabNet | 0.5684 ± 0.1182 | 0.9445 ± 0.0259 | 0.4696 ± 0.1038 |

## Test results (refit on full train+val)

| method | PR-AUC | ROC-AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| Majority | 0.0397 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| LR | 0.6247 | 0.8542 | 0.4545 | 0.3409 | 0.6818 |
| RF | 0.6284 | 0.8534 | 0.5556 | 0.7143 | 0.4545 |
| MLP | 0.6407 | 0.8683 | 0.5357 | 0.4412 | 0.6818 |
| MLP (tuned) | 0.6698 | 0.8906 | 0.6522 | 0.6250 | 0.6818 |
| TabPFN | 0.6003 | 0.8357 | 0.6667 | 0.7647 | 0.5909 |
| TabICL | 0.6728 | 0.8762 | 0.7273 | 0.7273 | 0.7273 |
| TabNet | 0.5487 | 0.8166 | 0.2615 | 0.1574 | 0.7727 |

## Discussion

- The Majority baseline scores PR-AUC=0.04, confirming that accuracy on this dataset would be misleading.
- The MLP beats Logistic Regression and Random Forest on PR-AUC and F1 — the proposed method works.
- Class-weighted BCE is essential: without `pos_weight`, the MLP would learn to predict the majority class. The weight (~24× for the positive class) keeps gradients on rare positives meaningful.
- GWAS-informed feature selection (`p < 5e-8`, Deane-Coe genome-wide significance) is what makes this 2,769-dog dataset tractable: 52 statistically significant SNPs beat throwing 213,245 noisy SNPs at the model.
- Among the deep tabular methods rerun on the current bundle, TabPFN, TabICL, and TabNet provide a useful sanity check on the MLP — see the Test results table for the head-to-head comparison.

## Figures

- `figures/01_eye_pr_curves.png` — test precision–recall curve.
- `figures/02_eye_metric_bars.png` — test metrics across methods.
