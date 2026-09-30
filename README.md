# Microbiome / GCMS Prediction App

A Streamlit application that predicts binary target values from ten participant
features. The bundled default dataset is
`Input_Data/GCMS_Metabolite_Data_Combined.xlsx` (39 participants, 286 targets).

## Setup

Use Python 3.10–3.13 (tested with Python 3.13):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an NVIDIA GPU on Linux or Windows WSL2, install the GPU dependencies instead:

```bash
python -m pip install -r requirements-gpu.txt
```

The NVIDIA driver must already work (`nvidia-smi`). The dependency setup follows
[TensorFlow's installation guide](https://www.tensorflow.org/install/pip).
`app/runtime.py` selects the GPU, enables memory growth, and handles the missing
cuSOLVER search path in the TensorFlow 2.21 Linux wheel. CPU is the fallback.
Restart an already running app after changing its environment.

## Build the default model once

On first launch, select **Default GCMS model** and click **Build default model**
in the expanded training controls. The bundled dataset is used automatically;
**Predict** becomes available after the build succeeds.

Alternatively, from the repository root:

```bash
.venv/bin/python app/train_default.py
```

This trains from `Input_Data/GCMS_Metabolite_Data_Combined.xlsx` and writes a
versioned model bundle under `app/models/default/`. It includes both networks,
feature and target names, fitted scalers, evaluation results, training settings,
a dataset SHA-256 fingerprint, and elapsed training time. The final saved model
is fitted on **all** training rows, after evaluation is complete.

Repeat runs reuse the existing bundle when the dataset, configuration, and
pipeline version match. Prediction never starts a training job. To rebuild:

```bash
.venv/bin/python app/train_default.py --force
```

Optional evaluation and training controls:

```bash
.venv/bin/python app/train_default.py --validation kfold
.venv/bin/python app/train_default.py --validation loo
.venv/bin/python app/train_default.py --validation none --epochs 500 --batch-size 32
```

`none` saves a model without reporting evaluation metrics. `loo` is intentionally
slow. The dataset and output directory can be overridden with `--data` and
`--output`. Generated bundles are local artifacts ignored by Git; run the build
command on a fresh checkout. Existing legacy files in `app/models/` are preserved.

## Run and predict

```bash
.venv/bin/python -m streamlit run app/ui.py --server.port 10880
```

Open the displayed URL, choose **Default GCMS model**, fill in the participant
form, and click **Predict**. The saved model loads automatically. Results show
one row per target and can be downloaded as CSV. Cancelling the download keeps
the last successful result and its submitted inputs visible for the current session.

To use another dataset, choose **Uploaded-data model**, expand the training
controls, upload an Excel file, and build the model. Uploaded-data models are
stored separately under `app/models/custom/`; they do not replace the default.
The UI also offers a default-model build button if no default bundle exists.

Use **Saved model build** to choose an earlier build for prediction. All successful
builds are retained on disk, including the last five and any older builds, separately
for default and uploaded-data models. The selector shows the build time (UTC),
evaluation method, epochs, batch size, and unique build ID. Reusing unchanged data
and settings does not create a new build; enable **Force retraining** to build again.
Selecting an older build does not replace the latest saved build. Results identify
the exact build used. Help icons beside evaluation method, epochs, and batch size
explain the training options.

## Performance and evaluation

The previous default performed leave-one-out validation, fitting two networks
for each participant and then fitting both again: **80 network fits for 39 rows**.
The new default uses a seeded 20% holdout (8 rows) followed by a full-data fit:
**4 network fits**. Five-fold and leave-one-out validation remain available.
Holdout scores and cross-validation scores measure different evaluations and
should not be compared as if they used the same test set.

Training uses float32 tensors, batches of 32, and a compiled in-memory epoch loop,
avoiding Keras input-pipeline setup on every epoch and repeated XLA compilation.
Each network independently stops after 25 epochs without sufficient training-loss
improvement and restores its best weights. The maximum remains 500 epochs.
Prediction calls the networks directly, avoiding `predict()` input-pipeline overhead.
The network architectures, losses, and regularization remain the same.

The saved default bundle records the measured runtime and metrics in
`training.json`. GPU warm-up and kernel compilation can affect runtime. A tiny
holdout is only a limited quality estimate; use cross-validation for a broader
assessment. No model-quality improvement is claimed from these speed changes.

## Input and output semantics

The Excel layout is: identifier column, ten numeric/participant features, then
numeric target columns. The feature list includes `Sex`, which accepts Male,
Female, 1, or 0. Encoding handles pandas object, string, and categorical columns,
ignoring surrounding whitespace and case. Prediction checks missing features,
reorders them to the saved training schema, and rejects nonnumeric/missing values.

The existing target definition is preserved: values above the **whole dataset's
global target mean** become 1, all others 0. This threshold defines the labels
before splitting; scalers and models are fitted only on each evaluation training
partition. Outputs are binary indicators, not predicted metabolite concentrations
or probabilities. Evaluation scores refer to those predefined binary labels.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover sex encoding, schema alignment, bad inputs, actual neural-network
learning, evaluation partitions, persistence, reuse, and failed rebuild behavior.
