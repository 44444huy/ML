# CLAUDE.md — Part B (Dog EVC)

You are working inside `dog/`, which is **Part B** of the FDP-EVC
project. Read this before doing anything in this folder.

## Scope

This is a beginner-level ML group project. Keep things simple. Do not
add interpretability tooling, ablation studies, or extra losses unless
the user explicitly asks for them.

## Hard rules

1. **Do not modify anything outside `dog/`.** Part A lives in
   `../human/` and is considered frozen unless the user explicitly asks
   otherwise.
2. **Do not import from Part A.** If you need a utility (metrics,
   splits helpers), copy it into `dog/src/` and adapt it. Part A and
   Part B must be independently runnable.
3. **All paths in `dog/` code resolve relative to `dog/`**, not to the
   repo root. Use `Path(__file__).resolve().parents[N]` carefully.
4. **Datasets live in `dog/data/raw/`.** Never commit raw genotype
   files (they are large and may have licence restrictions) — they go
   in `.gitignore`.

## Project context

The deliverable is a **proposed method** for predicting EVC traits from
DNA, validated on dog datasets where the labels are real ground truth
(unlike Part A, which used silver HIrisPlex-S labels). Two datasets:

- **Canine Eye GWAS** (~2,769 dogs, blue/brown, 3.9% positive) —
  implemented first because there is a published GWAS to drive
  feature selection.
- **Darwin's Ark coat color** (~3,277 dogs, multi-label coat) —
  implemented second.

## Proposed method (eye, current)

1. **GWAS-informed feature selection** — keep SNPs with `p_wald < 1.15e-7`
   (Bonferroni cutoff used by Deane-Coe et al. 2018 on this same
   dataset: 0.05 / ~430k markers). Yields ~60 SNPs, almost all on chr18
   in the ALX4 region. The `--top_k` flag is kept for legacy /
   sensitivity analysis only.
2. **MLP with class-weighted BCE** — `pos_weight = n_neg / n_pos` to
   handle the 4 % positive rate.
3. **PR-AUC + ROC-AUC + F1** as metrics (accuracy is meaningless at
   this imbalance).
4. **Compare against Majority / LR / RF baselines**.

That is the whole method. No focal loss, no SHAP, no ablation.

## Style

- Modular layout: `data/`, `models/`, `train/`, `evaluation/`.
- Persistent splits on disk (seed=42) so all methods are comparable.
- Every result JSON includes both CV-mean and held-out test metrics.
