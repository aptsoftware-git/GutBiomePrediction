# Microbiome improvement investigation

The Metabolite code and saved models were not changed in this investigation.
The Microbiome predictor was retained because the tested alternatives did not
produce a clear improvement across the relevant validation measures.

## Evaluation

The bundled file has 39 participants, ten predictors, and 1,638 targets.
Three shuffled five-fold splits (seeds 42, 7, 123) each evaluate all 39 participants.
These are three reuses of the same cohort, **not 117 independent participants**.

The comparison is between:

- The existing numeric predictor: four-fold inner selection of log1p ridge or a
  constant log-space baseline.
- Original-unit selection: four-fold inner selection of arithmetic mean, ridge
  (alpha 10/100/1000), partial least squares (1/2 components), Extra Trees
  (depth 3/6), or random forest (depth 3). Tree ensembles use 64 trees,
  minimum leaf size 2, 80% of features, seed 42. Original-unit squared error
  determines selection, emphasizing large abundance errors.
- The arithmetic-mean predictor, fitted separately inside each outer fold.

All feature scaling, model selection, and numeric bounds use only the current
training fold. Outer test participants are excluded from all those decisions.
All final predictions respect each fold's observed target ranges.

## Results pooled across the three splits

| Procedure | Numeric RMSE | Numeric MAE | Bacteroides RMSE |
| --- | ---: | ---: | ---: |
| Existing log-target selection | 3,267.19 | 139.51 | 80,933.42 |
| Original-unit model selection | 3,126.22 | 202.14 | 77,277.41 |
| Arithmetic-mean baseline | 3,119.61 | 202.16 | 77,069.31 |

The original-unit procedure lowers RMSE by about 4.3%, but increases MAE by about
44.9%. It does not beat the arithmetic mean overall on RMSE. Its inner folds
selected ridge or the mean, not PLS or tree ensembles. This suggests that the
modest RMSE change is largely a change in the prediction objective and typical
prediction level, rather than demonstrated improvement in patient-specific signal.
This is an interpretation of these development comparisons, not proof that the
features have no predictive information.

A separate exploratory screen evaluated distance-weighted nearest neighbours
(k=3/5) and RBF kernel ridge (alpha=1/10, gamma=0.1). It used the same three
five-fold splits with fixed settings, for both numeric targets and square-root
compositions. Nearest-neighbour numeric predictions had worse RMSE and MAE than
the existing predictor. Strongly regularized kernel ridge made a smaller RMSE/MAE
trade-off, not a clear improvement. None was adopted. A neighbour model can match
training rows very closely; that is not a reason to deploy it if held-out error
is worse.

## Product changes

Microbiome pipeline version 3 adds per-genus held-out numeric metrics to the saved
metadata and UI, plus a validation CSV download. It does **not** claim a new, more
accurate predictor. The report shows sample count, nonzero count, RMSE, MAE, R²,
baseline RMSE, and MSE improvement for every genus. Constant targets have undefined
R²; zero-error baselines have undefined relative improvement, shown as empty values.
Train-only builds do not show an independent validation report.

The default model was rebuilt with these diagnostics. Existing models remain
selectable. The same baseline, candidate set, fitted transformations, and bounds
are retained for the prediction algorithms. Previously generated predictions
should therefore be numerically unchanged for matching data and configuration.

## Reproduction and limitations

Raw results are in [microbiome_benchmark.json](microbiome_benchmark.json).
The file includes the source hash and split seeds. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python app/benchmark_microbiome.py --output /tmp/microbiome_benchmark.json
```

The benchmark does not publish or replace saved models. The fixed candidate
screening is exploratory; selecting a winner after inspecting these scores would
need independent confirmation. All comparisons use the same small development
cohort, not an external validation set. Numeric counts also retain sequencing-depth
variation that may not be inferable from the patient features.

A larger independent cohort, consistent laboratory processing, and useful additional
predictors are the next substantive avenues. No increase in epochs, forced exact-row
matching, or min/max clipping establishes better generalization to new participants.

References: [PLSRegression](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html),
[ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html),
[nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## User-proposed 1–10 range coding

A further experiment applied a separate `MinMaxScaler(feature_range=(1, 10))`
to each genus. Predictions remained continuous (no integer rounding), were
clipped to the coded range, inverse-transformed, and bounded in original units.
Training minima/maxima and feature scalers were fitted inside each training fold.
Constant targets, including all-zero genera, stayed constant after decoding.

The experiment compared a constant mean, ridge (alpha 1/10/100/1000/10000), and
shallow Extra Trees (depth 3) using inner-fold squared error in coded units.
On the same three outer split seeds, pooled numeric RMSE was **3,152.90** and
MAE **205.20**, compared with **3,267.19** and **139.51** for the existing predictor.
This again trades slightly lower RMSE for substantially higher absolute error
and does not establish an overall improvement. The deployed predictor was not
replaced by the range-coded alternative.

At a fixed alpha, affine target scaling and decoding leave ordinary independent
ridge predictions mathematically unchanged (with an intercept). The experiment's
separate fixed-alpha check found a maximum difference of about 0.018 original
units from floating-point computations. Changes in the selected model come from
the selection objective or model class, not from simply writing a large value as
a small number.

For Bacteroides, encoding is `1 + 9 * (value - 139) / 292512`, and decoding is
`139 + (code - 1) * 292512 / 9`. Values 139, 146395, and 292651 become 1, 5.5,
and 10. Rounding to integer codes would discard information; one code unit spans
about 32,501 original units.

Results: [microbiome_minmax_benchmark.json](microbiome_minmax_benchmark.json).
Reproduce without changing any saved model:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python app/benchmark_minmax.py --output /tmp/minmax.json
```

Reference: [scikit-learn MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
