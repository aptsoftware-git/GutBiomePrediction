# Application flow and models

## Training

1. `app/train_default.py` or the UI calls `train_controller()`.
2. The controller hashes the workbook and checks saved configuration and pipeline
   version. An unchanged, complete bundle is reused unless retraining is forced.
3. `loadandpreprocess.py` selects columns 1–10 as features and columns 11 onward
   as targets (zero-based). Shared feature preparation encodes Sex and validates
   numeric, finite inputs. Targets retain the existing whole-dataset mean
   binarization rule.
4. Evaluation runs on a fixed 20% holdout by default. Optional alternatives are
   five-fold validation, leave-one-out, or no evaluation. Each evaluation fit has
   independent scalers and models; test features do not enter their training.
5. A final independent fit uses all rows. Evaluation scores are not recomputed on
   the final training data.
6. Both models, scalers, schema, and provenance are saved in a new version
   directory. The `CURRENT` pointer is atomically replaced only after all files
   have been written. Failed training or saving leaves the previous version usable.

The default model lives in `app/models/default/`; the uploaded-data model lives
in `app/models/custom/`. Previous versions are retained. Legacy root-level model
files are left intact. Model artifacts use pickle for locally generated scaler
metadata and should only be loaded from trusted local training runs.

## Networks

The architecture is unchanged:

- Encoder: target inputs → Dense(512, ReLU, L2) → Dense(128, ReLU, L2)
  → linear latent vector (default dimension 10).
- Decoder: latent vector → Dense(128, ReLU) → Dense(512, ReLU)
  → linear target outputs.
- Autoencoder: encoder + decoder, Adam optimizer, Huber reconstruction loss.
- Predictor: participant features → Dense(64, ReLU, L2) → Dense(32, ReLU, L2)
  → linear latent outputs; Adam optimizer and MSE loss.

Each fit standardizes features and binary targets, learns the autoencoder, encodes
its training targets, then trains the predictor against those latent vectors.
`training.py` uses TensorFlow automatic differentiation and a compiled epoch loop,
including each network's regularization losses. It shuffles rows each epoch and
trains batches of up to 32. Defaults: at most 500 epochs, patience 25, minimum
training-loss improvement 1e-5. Best model weights are restored independently.
The seed is 42; GPU operations can still introduce numerical nondeterminism.

## Prediction

The controller resolves the selected bundle's `CURRENT` version, then loads and
caches the networks and scalers. Cache keys include the immutable version path,
so a newly published version is loaded without serving stale weights.

Shared preprocessing encodes Sex regardless of pandas dtype, selects and orders
the saved feature columns, and validates the input. The fitted feature scaler
transforms the row; predictor and decoder are called directly. The target scaler
is inverted and values above 0.5 become 1. Target names are restored, and the UI
shows the complete table and provides a CSV download. No training occurs here.

## Performance changes

With 39 rows, leave-one-out previously required 40 pairs of model fits. Default
holdout evaluation now requires only two pairs. The compiled training loop also
avoids per-epoch data-pipeline overhead and unnecessary XLA compilation per fold.
Float32 tensors stay in the training loop on the selected device. The UI identifies
the GPU, or warns when CPU fallback is active. See README for build commands,
evaluation limitations, and verification instructions.
