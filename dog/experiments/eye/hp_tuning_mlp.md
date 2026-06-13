# MLP Hyperparameter Tuning (5-fold CV on trainval)

**Grid size**: 72 configs × 5 folds = 360 trainings.
**Selection**: rank by CV-mean PR-AUC. Test set was NOT used for selection.

## Best config

| Param | Value |
|---|---|
| hidden | 64 |
| n_layers | 2 |
| dropout | 0.3 |
| lr | 0.001 |
| weight_decay | 0.0001 |

CV-mean PR-AUC = **0.7420** (± 0.1145)
Test PR-AUC = **0.6698**, ROC-AUC = 0.8906, F1 = 0.6522

## Top 10 configs by CV PR-AUC

| Rank | hidden | n_layers | dropout | lr | weight_decay | CV PR-AUC | CV ROC-AUC | CV F1 |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 1 | 64 | 2 | 0.3 | 1e-03 | 1e-04 | 0.7420 ± 0.1145 | 0.9276 | 0.4869 |
| 2 | 64 | 3 | 0.3 | 1e-03 | 1e-03 | 0.7413 ± 0.1241 | 0.9206 | 0.5090 |
| 3 | 64 | 3 | 0.3 | 1e-03 | 1e-04 | 0.7401 ± 0.1206 | 0.9208 | 0.4970 |
| 4 | 64 | 1 | 0.3 | 1e-03 | 1e-03 | 0.7400 ± 0.1268 | 0.9328 | 0.4959 |
| 5 | 64 | 2 | 0.3 | 1e-03 | 1e-03 | 0.7399 ± 0.1148 | 0.9262 | 0.4897 |
| 6 | 64 | 2 | 0.5 | 5e-04 | 1e-04 | 0.7398 ± 0.1233 | 0.9254 | 0.5397 |
| 7 | 256 | 3 | 0.5 | 5e-04 | 1e-03 | 0.7390 ± 0.1284 | 0.9174 | 0.5589 |
| 8 | 64 | 2 | 0.5 | 1e-03 | 1e-04 | 0.7386 ± 0.1246 | 0.9270 | 0.5799 |
| 9 | 64 | 2 | 0.5 | 5e-04 | 1e-03 | 0.7386 ± 0.1238 | 0.9247 | 0.5472 |
| 10 | 256 | 3 | 0.5 | 5e-04 | 1e-04 | 0.7383 ± 0.1287 | 0.9230 | 0.5491 |

## Worst 3 configs (sanity check)

| Rank | hidden | n_layers | dropout | lr | weight_decay | CV PR-AUC |
|---:|---:|---:|---:|---|---|---:|
| 70 | 128 | 3 | 0.3 | 1e-03 | 1e-03 | 0.7236 |
| 71 | 128 | 2 | 0.3 | 5e-04 | 1e-03 | 0.7225 |
| 72 | 128 | 1 | 0.3 | 5e-04 | 1e-04 | 0.7219 |
