# Metabolite & Microbiome Prediction

A Streamlit app with one participant form and two independently saved models:

| Prediction | Default training file | Output |
| --- | --- | --- |
| Metabolite | `Input_Data/GCMS_Metabolite_Data_Combined.xlsx` | 286 binary indicators |
| Microbiome | `Input_Data/Genus_Abundance_Fecal Data_Combined.xlsx` | 1,638 genus percentages plus bounded numeric values |

Both bundled files contain the same 39 participants and identical columns A–K.
Microbiome percentages sum to 100 before display rounding. The same table and CSV
also contain **Predicted value** in the original file's units. Training minima and
maxima are retained internally and omitted from prediction tables and CSVs. Numeric estimates use a separately trained predictor and
are clipped to each genus's observed training range; they need not reproduce the
percentage profile. Neither output measures absolute bacterial load.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/ui.py --server.port 10880
```

Streamlit 1.63 or later supports the numeric press-and-hold enhancement.
For an NVIDIA GPU, install `requirements-gpu.txt` instead; a working NVIDIA driver
is also required. TensorFlow uses the CPU when no GPU is available. Microbiome
regression uses scikit-learn on the CPU and does not require CUDA.

Choose **Default models**, expand **Build or update saved models**, and click
**Build / reuse both models**. Individual build buttons update either model.
Choose a saved build for each prediction, enter the participant once, and click
**Predict**. Both models must be ready. Each result has its own CSV download.
Downloads and widget reruns preserve the last successful pair; a prediction
failure never combines a new result with an old result. The displayed input and
model identities always belong to that saved pair.

## Training from the command line

```bash
.venv/bin/python app/train_default.py
.venv/bin/python app/train_default.py --task microbiome
.venv/bin/python app/train_default.py --task metabolite --validation kfold
.venv/bin/python app/train_default.py --force
```

The default command builds or reuses **both** models. Metabolite evaluation defaults
to a seeded 20% holdout; microbiome evaluation defaults to nested five-fold
cross-validation. Each task supports `holdout`, `kfold`, `loo`, and `none`.
`--validation`, `--epochs`, and `--batch-size` apply to metabolites.
`--microbiome-validation` controls microbiome outer evaluation. Even with `none`,
microbiome model selection uses inner cross-validation; no independent evaluation
score is reported in that case. Final models are fitted on every training row.

`--data` / `--output` override the metabolite file/directory;
`--microbiome-data` / `--microbiome-output` override the microbiome equivalents.

## Uploaded datasets

Choose **Uploaded-data models**. Separate upload controls accept Metabolite and
Microbiome Excel files. Building both requires both files; an individual build
requires only its corresponding file. An existing saved model of the other type
can be reused. The files may contain different participants because the models
are trained independently, but feature names, units, and intake periods must agree.

Excel layout:

- Column A: a unique, nonempty participant identifier (not a predictor).
- Columns B–K: Age, Sex, Height (cm), Weight (Kgs), BMI, Body fat percentage,
  MUAC (cm), Protein (grams), Carbohydrates  (grams), Fats (grams).
- Columns L onward: numeric targets, with a unique name for each target.

New training and the UI use **Weight (Kgs)**. Legacy `Weight (cm)` and
`Weight (kg)` headers are accepted aliases. Older saved models receive their
original feature key automatically, so existing metabolite models still work. `Carbohydrates (grams)` is also accepted. Sex accepts Male/Female or 1/0.
Missing, infinite, and nonnumeric data are rejected. Microbiome abundances must
be nonnegative, with a positive total per row. Counts, proportions, or percentages
are normalized per participant; no sequencing-depth input is needed. Numeric microbiome predictions retain the original supplied units; use consistent
units across participants. Microbiome training requires at least six independent participants. Repeated measurements
from one person should not be supplied as independent rows.

## Saved model compatibility

| Task | Default directory | Uploaded directory |
| --- | --- | --- |
| Metabolite | `app/models/default/` | `app/models/custom/` |
| Microbiome | `app/models/microbiome/default/` | `app/models/microbiome/custom/` |

Existing metabolite bundles and histories remain usable without migration.
Every successful build has a unique version, training configuration, source hash,
metrics, and timestamp. Matching data/configuration/pipeline versions reuse the
current model. All versions are retained. Atomic publication updates `CURRENT`
only after all artifacts are written; failed rebuilds preserve the previous model.
Generated artifacts are ignored by Git, so build them on a fresh checkout.
Only load locally trusted saved artifacts; the upload controls accept data, not
serialized models.

## Model behavior and limitations

Metabolite prediction retains the existing autoencoder + latent predictor and
binary label definition: targets above the dataset-wide global mean become 1;
the remainder become 0. These outputs are not concentrations or probabilities.
The threshold is defined before splitting; training scalers and networks fit only
on the training partition during evaluation.

Microbiome prediction compares a constant population profile with ridge regression
on Hellinger-transformed or centered-log-ratio-transformed compositions. It tunes
regularization in inner folds and evaluates the complete selection procedure on
held-out outer folds. It preserves all genera and returns nonnegative percentages.
See [MICROBIOME_MODEL.md](MICROBIOME_MODEL.md) for the rationale, validation details,
and measured results on the bundled data.

With 39 participants and 1,638 targets, results are exploratory. The UI reports
held-out metrics and baseline comparison, identifies a selected constant model,
and flags numeric inputs outside observed training ranges. Neither validation
nor numeric precision establishes accurate individual-genus prediction.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover both model types, fold isolation, relative-abundance semantics,
validation, persistence/reuse/failure, dual uploads, version selection, and paired
results/downloads. Optional `pip install quickjs` enables the numeric-control
JavaScript event tests; they use a simulated DOM, not a full browser.

## Numeric microbiome predictions and ranges

Each genus's observed minimum and maximum are saved with its trained numeric
predictor. The default Bacteroides range is **139–292,651**. Numeric prediction uses
log1p-target ridge regression selected against a constant baseline in nested
validation; both selection and outer evaluation apply bounds learned only from
the corresponding training rows. Final bounds use all rows in the selected build's
training file. These are observed ranges, **not confidence intervals**.

Pipeline version 2 saves both microbiome predictors together. Older microbiome
versions remain selectable for percentages; rebuild a model to add numeric values. Uploads receive their own learned bounds. The CSV includes the
same numeric predictions as the table, at full precision.

## Microbiome accuracy reports

New Microbiome builds include **Microbiome accuracy by genus** with held-out
RMSE, MAE, R², baseline comparison, and a downloadable validation CSV. This report
is separate from the two prediction downloads and is unavailable for train-only
builds. Negative scores and poor baseline comparisons identify limitations; they
are not hidden by averaging across genera.

A comparison of additional regressors did not establish a clearly better predictor
for the bundled cohort. See [the benchmark results](reports/MICROBIOME_BENCHMARK.md).
Microbiome pipeline version 3 preserves the prediction algorithms and adds these
diagnostics. Metabolite training and saved artifacts are unaffected.
