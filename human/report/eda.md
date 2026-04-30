# EDA Summary

**N = 2481 samples, 41 SNPs**


## Class distribution (argmax labels)

- **eye**: blue=1438 (58.0%), intermediate=0 (0.0%), brown=1043 (42.0%)
- **hair**: blond=0 (0.0%), brown=150 (6.0%), red=4 (0.2%), black=0 (0.0%), light=816 (32.9%), dark=1511 (60.9%)
- **skin**: very_pale=8 (0.3%), pale=1 (0.0%), intermediate=1114 (44.9%), dark=157 (6.3%), dark_to_black=1201 (48.4%)

## Low-confidence rates

| threshold | eye | hair | skin |
|---|---|---|---|
| 0.5 | 0.44 | 0.00 | 5.24 |
| 0.6 | 61.19 | 5.84 | 12.45 |
| 0.7 | 72.23 | 22.49 | 21.77 |
| 0.8 | 76.30 | 60.98 | 33.90 |

## Key observations

1. **Argmax collapses classes.** Eye `intermediate`, Hair `blond`/`black`, Skin `pale` almost never win argmax, though HIrisPlex assigns them non-trivial probabilities.
2. **Eye confidence is poor**: majority of samples have max p_value < 0.6.
3. **Hair / Skin are imbalanced**: one class dominates (>60%).
4. **SNPs show correlation blocks** — expected from linkage disequilibrium (neighboring SNPs inherited together).