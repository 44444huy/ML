# Part B — EVC on Dogs

This is the dog-dataset half of the EVC project. It is intentionally
**isolated from Part A** (`../src/`, `../data/`, `../experiments/`,
`../report/`) so that:

- Part A (humans, HIrisPlex silver labels) keeps its own pipeline and
  results untouched.
- Part B (dogs, true ground-truth phenotypes) can use a different SNP
  representation, different label format, and different metrics without
  polluting Part A.

## Why dogs

Part A established a clean baseline on the human HIrisPlex dataset and
exposed five structural problems caused by training on argmax(silver
label). The most fundamental of those — *no real ground truth* — cannot
be fixed inside Part A. Part B side-steps it by using two dog datasets
where phenotype is observed directly:

1. **Darwin's Ark coat color** — ~3,277 dogs with genome-wide SNPs and
   self-reported coat colors (multi-label: black, white, red, …).
   *Done first* — labels are clean, classes are reasonably balanced.
2. **Canine Eye Color GWAS** — ~2,769 dogs with blue/brown eye labels.
   *Done second* — rare-event problem (3.9% blue), needs different
   handling (focal loss, PR-AUC).

## Layout

```
dog/
├── data/
│   ├── raw/         # .bed/.bim/.fam + phenotype CSVs (gitignored)
│   └── processed/   # npz/parquet bundles (gitignored)
├── src/
│   ├── data/        # preprocess_coat.py, preprocess_eye.py, splits.py
│   ├── models/
│   ├── losses/
│   ├── train/
│   └── evaluation/  # metrics adapted for multi-label + rare-event
├── experiments/
│   ├── coat/
│   └── eye/
└── report/
    ├── figures/
    ├── coat.md
    └── eye.md
```

## Status

- [ ] Download datasets (waiting on links)
- [ ] Preprocess coat-color dataset
- [ ] Baselines on coat color
- [ ] Proposed method on coat color
- [ ] Preprocess eye-color dataset
- [ ] Baselines on eye color (rare-event)
- [ ] Proposed method on eye color
- [ ] Final report
