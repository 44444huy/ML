# MLP Hyperparameter Tuning (5-fold CV on trainval)

**Grid size**: 72 configs × 5 folds = 360 trainings.  
**Selection**: rank by CV-mean PR-AUC. Test set was NOT used for selection.

## Best config

| Param | Value |
|---|---|
| hidden | 128 |
| n_layers | 1 |
| dropout | 0.3 |
| lr | 0.0005 |
| weight_decay | 0.0001 |

CV-mean PR-AUC = **0.7474** (± 0.1376)  
Test PR-AUC = **0.6564**, ROC-AUC = 0.9134, F1 = 0.6667

## Top 10 configs by CV PR-AUC

| Rank | hidden | n_layers | dropout | lr | weight_decay | CV PR-AUC | CV ROC-AUC | CV F1 |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 1 | 128 | 1 | 0.3 | 5e-04 | 1e-04 | 0.7474 ± 0.1376 | 0.9441 | 0.5194 |
| 2 | 128 | 1 | 0.3 | 1e-03 | 1e-04 | 0.7473 ± 0.1350 | 0.9429 | 0.5406 |
| 3 | 128 | 1 | 0.3 | 5e-04 | 1e-03 | 0.7472 ± 0.1376 | 0.9449 | 0.5260 |
| 4 | 128 | 1 | 0.3 | 1e-03 | 1e-03 | 0.7465 ± 0.1352 | 0.9437 | 0.5410 |
| 5 | 128 | 1 | 0.5 | 1e-03 | 1e-04 | 0.7463 ± 0.1335 | 0.9428 | 0.5333 |
| 6 | 128 | 1 | 0.5 | 1e-03 | 1e-03 | 0.7455 ± 0.1339 | 0.9435 | 0.5456 |
| 7 | 64 | 3 | 0.5 | 1e-03 | 1e-03 | 0.7432 ± 0.1263 | 0.9305 | 0.6289 |
| 8 | 128 | 3 | 0.3 | 5e-04 | 1e-04 | 0.7420 ± 0.1224 | 0.9224 | 0.6592 |
| 9 | 64 | 3 | 0.3 | 1e-03 | 1e-03 | 0.7415 ± 0.1267 | 0.9278 | 0.5489 |
| 10 | 128 | 1 | 0.5 | 5e-04 | 1e-04 | 0.7414 ± 0.1353 | 0.9432 | 0.5409 |

## Worst 3 configs (sanity check)

| Rank | hidden | n_layers | dropout | lr | weight_decay | CV PR-AUC |
|---:|---:|---:|---:|---|---|---:|
| 70 | 64 | 1 | 0.5 | 5e-04 | 1e-04 | 0.7221 |
| 71 | 64 | 1 | 0.3 | 5e-04 | 1e-03 | 0.7218 |
| 72 | 64 | 1 | 0.5 | 5e-04 | 1e-03 | 0.7213 |
