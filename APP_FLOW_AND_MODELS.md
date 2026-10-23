# Application flow and saved models

`app/ui.py` offers a shared participant form and two model tasks. Each task has
its own default/uploaded training source, configuration, version selector, and
artifact directory. Both models must exist before Predict is enabled.

1. Select Default models or Uploaded-data models.
2. Build both models, or build either one individually. For uploads, supply the
   file for each requested task. Existing bundles are reused when hashes,
   configurations, and pipeline versions match.
3. Choose a saved version independently for each task.
4. Submit one participant row. The same row is passed to both selected bundles.
5. Publish both result tables to session state only after both predictions succeed.
6. Render separate Metabolite and Microbiome tables and CSV downloads. Keep the
   last successful pair and its provenance across reruns and failed submissions.

## Implementation map

- `loadandpreprocess.py`: shared feature validation and aliases; binary metabolite
  labels or unthresholded nonnegative microbiome targets.
- `main.py`: existing neural metabolite training, both training controllers,
  versioned artifact publication, cache, loading, and prediction dispatch.
- `microbiome.py`: relative-abundance transformations, regularized multi-output
  regression, constant baselines, nested selection/evaluation, and an independently
  fitted original-unit predictor with per-genus training bounds.
- `training.py`, `encoder.py`, `predictor.py`, `runtime.py`: existing TensorFlow
  metabolite model and device configuration.
- `train_default.py`: build both defaults or one selected task from the CLI.
- `static/number_input_hold.html`: delegated numeric step-button hold behavior.

Metabolite bundles retain `predictor.keras`, `decoder.keras`, `meta.pkl`, and
`training.json`. Microbiome bundles use `bundle.pkl` and `training.json` with a
`task: microbiome` marker. Microbiome pipeline version 2 saves percentage and
bounded numeric predictors together, with per-target ranges in the metadata. Default/uploaded directories are separate for each task;
see README. Existing metabolite artifacts remain readable. Each model's `CURRENT`
pointer updates only after that model is fully saved. Training the two models is
independent: if one fails, the successful build remains available and the failed
model's previous version remains intact.

See [MICROBIOME_MODEL.md](MICROBIOME_MODEL.md) for the statistical method and its
limitations and [README.md](README.md) for usage.
