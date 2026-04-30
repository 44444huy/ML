# FDP-EVC — Externally Visible Characteristics from DNA

A two-part student project on predicting visible traits (eye, hair, skin /
coat colour) from genotype data.

```
FDP-EVC/
├── human/   ← Part A: HIrisPlex-style pipeline on humans (silver labels)
└── dog/     ← Part B: same problem on dogs, with real ground-truth labels
```

## Part A — Humans (silver labels)

**Status: complete.**

Dataset: 2,481 individuals, 41 HIrisPlex SNPs, labels obtained by running
the published HIrisPlex-S logistic-regression model. Because the labels
are produced by another model rather than observed directly, they are
*silver*, not ground truth.

Part A:
1. Establishes a clean baseline (LR / SVM / RF) with persistent splits
   and a unified metric suite.
2. Exposes five structural problems that come from training on
   argmax(silver label):
   - Eye `intermediate` class is wiped out by argmax.
   - Many low-confidence samples become noisy hard labels.
   - Severe class imbalance (red hair, very-dark skin).
   - Hair / skin are ordinal but treated as categorical.
   - Three traits trained independently, ignoring biological
     correlation.
3. Proposes **soft-label KL distillation** (M2) as a first fix and shows
   it beats every baseline on skin (macro-F1 0.774 vs 0.652, MAE 0.022
   vs ~0.11).

See `human/report/` for the full write-up.

## Part B — Dogs (ground-truth labels)

**Status: in progress.**

The fundamental limit of Part A is the silver label itself. Part B
side-steps it by using two dog datasets where phenotype is observed
directly, not predicted by another model:

1. **Coat colour** — Darwin's Ark (~3,277 dogs, multi-label). Owner
   self-reports through a structured survey (Q243). Implemented first.
2. **Eye colour** — Deane-Coe et al. (~2,769 dogs, blue vs brown,
   3.9 % positive). Phenotype from owner-uploaded photos verified by
   commercial DNA-testing staff. Rare-event problem, implemented
   second.

See `dog/README.md` for the Part B status and layout.

## Why two parts

Part A and Part B are both legitimate exercises but they answer
different questions:

| | Part A (human) | Part B (dog) |
|---|---|---|
| Label source | HIrisPlex-S model output | Owner survey / photo |
| Label type | Silver | Ground truth (with noise) |
| Question | What goes wrong on silver labels? | What's possible with real labels? |

The narrative for the final defense is "Part A shows the ceiling
imposed by silver labels; Part B shows what changes when the ceiling
is removed".

## Repository conventions

- Each part is self-contained — no cross-imports between `human/` and
  `dog/`.
- Persistent splits with seed 42 inside each part.
- Raw data is gitignored; processed bundles are gitignored.
- Each part has its own `README.md` and (for `dog/`) its own
  `CLAUDE.md` with rules for AI-assisted development.
