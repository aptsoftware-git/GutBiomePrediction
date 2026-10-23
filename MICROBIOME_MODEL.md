# Microbiome model rationale and validation

## Dataset and prediction target

The bundled genus file has 39 independent participant rows, ten predictors, and
1,638 genus columns. Approximately 72.4% of abundance entries are zero; no genus
is globally all zero. Row totals range from 24,362 to 891,358. The first eleven
columns exactly match the bundled metabolite file.

One output is **relative abundance (%) per genus**. Raw counts cannot be
interpreted as absolute microbial loads from these data. Per-row normalization
removes sequencing-total differences; the predicted profile sums to 100 across
the genera included in the training file. No read-depth input is needed.
See [Gloor et al., 2017](https://pubmed.ncbi.nlm.nih.gov/29187837/) on the
compositional interpretation of microbiome sequencing data.

## Candidate models

A second neural network is not the default: 39 participants provide little evidence
to estimate a complex mapping to 1,638 targets. We compare a small, predefined set:

1. A constant population profile: the normalized square of the average
   square-root composition (the Hellinger centroid).
2. Multi-output ridge regression on square-root proportions (Hellinger transform).
3. Multi-output ridge regression on centered log ratios, using a fixed relative
   pseudocount of `1e-6` for zero values.

Each ridge method tests alpha values 1, 10, 100, 1,000, and 10,000. The ten feature
columns are standardized using only the current training partition. Ridge shrinks
coefficients and is computationally modest for this small feature matrix;
[scikit-learn's Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
describes the objective and multi-output support. This candidate set is a practical
starting point, not evidence of a globally best algorithm.

For Hellinger predictions, negative transformed values are clipped to zero,
squared, and normalized. Log-ratio predictions use a stable softmax. Targets never
observed in a fit's training partition stay zero. A degenerate zero predicted
profile falls back to that training partition's constant profile. Every supplied
genus remains in the output. A zero estimate is not evidence of biological absence.

## Selection without evaluation leakage

Default evaluation uses shuffled five-fold outer cross-validation (seed 42).
Within each outer training partition, four-fold inner cross-validation chooses the
transform and regularization, or the constant baseline. Scaling, coefficients,
zero-target masks, and baseline profiles use training rows only. Inner selection
minimizes mean squared Hellinger distance between predicted and observed profiles.
A constant baseline is itself a candidate, allowing the procedure to decline a
patient-specific regression if it adds no validated benefit.

The baseline uses the same loss geometry as the regressions: comparing only with
an arithmetic average would confound a better constant representation with useful
patient information. Baseline error is computed separately on each outer test fold.
After evaluation, inner selection runs again on all rows and the chosen model is
fitted on all rows. Its inner score is **not** reported as independent performance.
See [scikit-learn's nested cross-validation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Repeated measurements, household clustering, or batch effects would need group-aware
validation and suitable metadata; the supplied files identify one row per participant.
No uncertainty intervals are claimed from 39 rows. No targets are filtered based on
the held-out data, and no dimensionality reduction is fitted on the full dataset
before evaluation.

## Measured bundled-data results

Local default build, seed 42, five outer folds, 39 held-out participant predictions:

| Measure | Value |
| --- | ---: |
| Final selected method | Hellinger ridge |
| Final alpha | 100 |
| Mean squared Hellinger distance | 0.274361 |
| Constant-profile baseline distance | 0.291628 |
| Relative error reduction against baseline | 5.92% |
| RMSE across genus percentages | 0.827974 percentage points |
| MAE across genus percentages | 0.067210 percentage points |
| Mean per-genus R², nonconstant targets | -0.056007 |

Every outer fold selected Hellinger ridge with alpha 100. The modest improvement
in a whole-profile composition metric does **not** establish reliable prediction
of individual genera. Negative average genus R² and the small sample size are
material limitations. Average errors are also influenced by the many rare genera.
These are cross-validation estimates, not independent-cohort validation or proof
of clinical accuracy. New independent participants and external validation are
needed before drawing strong conclusions.

Scores, fold selections, baseline errors, and full-data inner candidate scores are
saved in each version's `training.json`. Uploaded datasets are independently
selected and evaluated; they do not inherit the bundled dataset's selected alpha.

## Additional numeric prediction (pipeline version 2)

Each build now also fits a separate original-unit predictor. It compares a constant
log-space mean against multi-output ridge regression on `log1p` targets, using the
same candidate alpha values and nested fold structure. It selects by log1p MSE.
Each fit learns its own per-genus minimum and maximum solely from its training
rows. Predictions are clipped to these bounds, including inside inner selection
and outer evaluation. The final predictor's bounds come from the complete training
file and are stored both in the bundle and in `training.json`.

The output table and CSV retain percentages and add numeric predictions, training
minima, and training maxima. Constant targets remain constant, and all-zero targets
remain zero. Numeric predictions are fractional estimates, not rounded observed
read counts. These are predictions in the original file's units, not estimates of
absolute microbial load. The independently fitted numeric and percentage outputs
are not required to imply an identical composition. Ranges are observed training
extrema rather than uncertainty or reference intervals.

For Bacteroides the bundled file's actual minimum is **139** and maximum is
**292,651**. For the default five-fold numeric model, nested validation produced
log1p MSE **0.808296** versus **0.814783** for the baseline (about **0.80%** improvement),
RMSE **3269.7121**, MAE **139.7874** in original units, and average per-genus R²
**-0.070931**. Thus bounded values should not be confused with accurate values;
individual numeric prediction remains weak with the available sample size.

## Improvement investigation and diagnostics (pipeline version 3)

Further comparisons across three five-fold splits did not establish a clearly
better alternative. Original-unit ridge, PLS, and tree ensemble selection reduced
RMSE modestly but increased MAE and did not consistently beat an arithmetic-mean
baseline. Fixed nearest-neighbour and kernel candidates also failed to provide a
clear overall improvement. The predictor algorithms are retained.

Version 3 adds a per-genus numeric validation report to the UI and saved metadata;
it is a diagnostic update, not an accuracy claim. See the full
[benchmark report](reports/MICROBIOME_BENCHMARK.md) and its reproducible script.
