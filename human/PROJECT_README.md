# EVC — Forensic DNA Phenotyping

Predict eye, hair, and skin colour from 41 HIrisPlex-S SNPs.

## Project layout

```
FDP-EVC/
├── hirisplex_results_FN_v2.csv   # raw HIrisPlex output (2504 subjects)
├── full_dataset.csv              # repo-provided flat dataset
├── data/processed/               # parsed npz + persistent splits
├── src/
│   ├── data/
│   │   ├── preprocess.py         # parse SNP vectors, build soft/hard labels
│   │   ├── eda.py                # figures + EDA summary
│   │   └── splits.py             # 80/20 holdout + stratified 5-fold CV
│   ├── models/
│   │   └── baselines.py          # B1 LR, B2 SVM-RBF, B3 Random Forest
│   └── evaluation/
│       ├── metrics.py            # accuracy, macro-F1, AUC, ECE, MAE, QWK
│       └── report_baselines.py   # renders baseline summary table + CMs
├── experiments/baselines/        # baseline_results.json
├── report/
│   ├── eda.md                    # EDA writeup
│   ├── baselines.md              # baseline results writeup
│   └── figures/                  # all PNGs
└── PROJECT_README.md
```

## Reproducing Week 1

```bash
# 1. parse raw HIrisPlex output into a clean dataset
python src/data/preprocess.py

# 2. build persistent train/test/CV splits
python src/data/splits.py

# 3. EDA
python src/data/eda.py

# 4. baselines
python src/models/baselines.py

# 5. render baseline report
python src/evaluation/report_baselines.py
```

## Week 1 findings

- N = 2481 subjects × 41 SNPs
- Argmax on HIrisPlex p_values **collapses rare classes**: eye `intermediate`, hair `blond`/`black`, skin `pale` essentially never appear as labels, even though HIrisPlex assigns them non-trivial probabilities.
- Eye confidence is weak: **61% of samples have max p_value < 0.6**.
- Baselines (LR, SVM, RF): accuracy 92–100%, but **macro-F1 only 0.42–0.67** — accuracy is misleading.
- Baselines by construction cannot beat HIrisPlex; they map 41 SNPs back to HIrisPlex's own output.

## Roadmap

- **Week 2** Soft-label MLP (M1): MLP trained with argmax labels — the deep-learning counterpart of B1-B3, to isolate the architecture effect from the label-treatment effect.
- **Week 3** Soft-label distillation (M2): KL loss against HIrisPlex p_values directly. Expected to lift macro-F1 on eye (recovers `intermediate`).
- **Week 4** Multi-task (M3) and ordinal heads (M4): shared backbone + CORAL loss for hair/skin. Expected to lower MAE.
- **Week 5** MC-Dropout uncertainty (M5); SHAP sanity check against known SNPs (rs12913832 for eye, rs16891982 for skin, rs1805007 for red hair); cross-population tests.
- **Week 6** Final report + slides.
