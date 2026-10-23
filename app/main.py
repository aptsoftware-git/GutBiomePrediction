"""Train, persist and reuse model bundles without shared mutable training state."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler

from runtime import tf, TRAINING_DEVICE, DEVICE_NAME
from encoder import encoder
from predictor import predictor
from loadandpreprocess import DEFAULT_DATA_PATH, MICROBIOME_DATA_PATH, loadandpreprocess, prepare_features
from microbiome import MicrobiomeBundle, MicrobiomeConfig, train_microbiome
from training import fit_in_memory

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_DIR = MODEL_DIR / "default"
CUSTOM_MODEL_DIR = MODEL_DIR / "custom"
DEFAULT_MICROBIOME_MODEL_DIR = MODEL_DIR / "microbiome" / "default"
CUSTOM_MICROBIOME_MODEL_DIR = MODEL_DIR / "microbiome" / "custom"
PIPELINE_VERSION = 2
MICROBIOME_PIPELINE_VERSION = 3


@dataclass(frozen=True)
class TrainingConfig:
    latent_dimension: int = 10
    epochs: int = 500
    batch_size: int = 32
    patience: int = 25
    validation: str = "holdout"
    seed: int = 42

    def __post_init__(self):
        for name in ("latent_dimension", "epochs", "batch_size", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.validation not in {"holdout", "kfold", "loo", "none"}:
            raise ValueError("Unknown validation method.")


@dataclass
class ModelBundle:
    predictor: object
    decoder: object
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    feature_columns: list
    target_columns: list

    def predict(self, frame):
        features = prepare_features(frame, self.feature_columns)
        scaled = self.x_scaler.transform(features).astype("float32")
        with tf.device(TRAINING_DEVICE):
            latent = self.predictor(scaled, training=False)
            decoded = self.decoder(latent, training=False).numpy()
        values = self.y_scaler.inverse_transform(decoded)
        return pd.DataFrame((values > 0.5).astype(int), columns=self.target_columns, index=frame.index)


def load_data(file_path=DEFAULT_DATA_PATH):
    return loadandpreprocess(file_path)


def _fit_models(X, Y, config, progress=None):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x = x_scaler.fit_transform(X).astype("float32")
    y = y_scaler.fit_transform(Y).astype("float32")
    options = dict(epochs=config.epochs, batch_size=config.batch_size,
                   patience=config.patience, progress=progress)
    with tf.device(TRAINING_DEVICE):
        auto, enc, dec = encoder(Y.shape[1], config.latent_dimension)
        if progress:
            progress("Learning target representation")
        auto_history = fit_in_memory(auto, y, y, **options)
        latent = enc(y, training=False)
        pred = predictor(X.shape[1], config.latent_dimension)
        if progress:
            progress("Learning participant-to-target prediction")
        pred_history = fit_in_memory(pred, x, latent, **options)
    bundle = ModelBundle(pred, dec, x_scaler, y_scaler, list(X.columns), list(Y.columns))
    return bundle, {"autoencoder": auto_history, "predictor": pred_history}


def model_training(X, Y, config=None, progress=None):
    config = config or TrainingConfig()
    tf.keras.utils.set_random_seed(config.seed)
    indexes = np.arange(len(X))
    if config.validation == "holdout":
        train, test = train_test_split(indexes, test_size=max(2, int(np.ceil(len(X) * .2))), random_state=config.seed)
        splits = [(train, test)]
    elif config.validation == "kfold":
        if len(X) < 6:
            raise ValueError("K-fold validation requires at least 6 samples.")
        splits = list(KFold(n_splits=min(5, len(X) // 2), shuffle=True, random_state=config.seed).split(X))
    elif config.validation == "loo":
        splits = list(LeaveOneOut().split(X))
    else:
        splits = []
    predictions, actual = [], []
    for fold, (train, test) in enumerate(splits, 1):
        if progress:
            progress(f"Evaluation {fold}/{len(splits)}")
        candidate, _ = _fit_models(X.iloc[train], Y.iloc[train], config, progress)
        predictions.append(candidate.predict(X.iloc[test]).to_numpy())
        actual.append(Y.iloc[test].to_numpy())
        del candidate
    metrics = {"validation": config.validation, "evaluation_samples": 0, "r2": None, "rmse": None}
    if predictions:
        truth, predicted = np.concatenate(actual), np.concatenate(predictions)
        metrics.update(evaluation_samples=len(truth),
                       r2=float(r2_score(truth, predicted, multioutput="uniform_average")),
                       rmse=float(np.sqrt(mean_squared_error(truth, predicted))))
    if progress:
        progress(f"Fitting saved model on all {len(X)} samples")
    bundle, history = _fit_models(X, Y, config, progress)
    return bundle, metrics, history


def _artifact_directory(model_dir, version=None):
    model_dir = Path(model_dir)
    pointer = model_dir / "CURRENT"
    if version is None and not pointer.exists():
        return None
    name = version if version is not None else pointer.read_text().strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("Invalid model version pointer.")
    directory = model_dir / name
    if not (directory / "training.json").exists():
        return None
    info = json.loads((directory / "training.json").read_text())
    files = ("bundle.pkl",) if info.get("task") == "microbiome" else ("predictor.keras", "decoder.keras", "meta.pkl")
    return directory if all((directory / f).exists() for f in files) else None


def model_info(model_dir=DEFAULT_MODEL_DIR, version=None):
    directory = _artifact_directory(model_dir, version)
    return json.loads((directory / "training.json").read_text()) if directory else None


def list_models(model_dir=DEFAULT_MODEL_DIR):
    """Return all complete saved builds, newest first; never delete older builds."""
    builds = []
    for directory in Path(model_dir).glob("run-*"):
        try:
            info = model_info(model_dir, directory.name)
            if isinstance(info, dict) and "trained_at" in info:
                builds.append({"version": directory.name, "info": info})
        except (OSError, ValueError):
            continue
    return sorted(builds, key=lambda build: (build["info"]["trained_at"], build["version"]), reverse=True)


def _save_trained_artifacts(bundle, info, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    # Publish the pointer last so readers never load half of a new model bundle.
    version = "run-" + uuid4().hex
    with TemporaryDirectory(prefix=".training-", dir=model_dir) as staging:
        destination = Path(staging)
        if isinstance(bundle, MicrobiomeBundle):
            with (destination / "bundle.pkl").open("wb") as handle:
                pickle.dump(bundle, handle)
        else:
            bundle.predictor.save(destination / "predictor.keras")
            bundle.decoder.save(destination / "decoder.keras")
            with (destination / "meta.pkl").open("wb") as handle:
                pickle.dump({"x_scaler": bundle.x_scaler, "y_scaler": bundle.y_scaler,
                             "feature_columns": bundle.feature_columns,
                             "target_columns": bundle.target_columns}, handle)
        (destination / "training.json").write_text(json.dumps(info, indent=2))
        os.replace(destination, model_dir / version)
    temporary_pointer = model_dir / (".CURRENT-" + uuid4().hex)
    temporary_pointer.write_text(version)
    os.replace(temporary_pointer, model_dir / "CURRENT")


@lru_cache(maxsize=4)
def _load_bundle(directory):
    directory = Path(directory)
    if (directory / "bundle.pkl").exists():
        with (directory / "bundle.pkl").open("rb") as handle:
            return pickle.load(handle)
    with (directory / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    with tf.device(TRAINING_DEVICE):
        pred = tf.keras.models.load_model(directory / "predictor.keras", compile=False)
        dec = tf.keras.models.load_model(directory / "decoder.keras", compile=False)
    return ModelBundle(pred, dec, **meta)


def train_controller(file_path=DEFAULT_DATA_PATH, *, model_dir=DEFAULT_MODEL_DIR,
                     config=None, progress=None, force=False):
    config = config or TrainingConfig()
    file_path = Path(file_path)
    source_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    existing = model_info(model_dir)
    if not force and existing and existing.get("source_sha256") == source_hash and existing.get("config") == asdict(config) and existing.get("pipeline_version") == PIPELINE_VERSION:
        if progress:
            progress("Reusing saved model; dataset and training settings have not changed.")
        return existing
    started = time.perf_counter()
    X, Y = load_data(file_path)
    bundle, metrics, history = model_training(X, Y, config, progress)
    info = {"pipeline_version": PIPELINE_VERSION, "source": file_path.name,
            "source_sha256": source_hash, "config": asdict(config), "metrics": metrics,
            "history": history, "samples": len(X), "targets": Y.shape[1],
            "device": DEVICE_NAME, "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_seconds": round(time.perf_counter() - started, 3)}
    _save_trained_artifacts(bundle, info, model_dir)
    return info


def predict_controller(user_input_row, model_dir=DEFAULT_MODEL_DIR, *, version=None):
    directory = _artifact_directory(model_dir, version)
    if directory is None:
        raise ValueError("No saved model is available. Build the default model or train on an uploaded file first.")
    return _load_bundle(str(directory.resolve())).predict(user_input_row)


def train_microbiome_controller(file_path=MICROBIOME_DATA_PATH, *,
                                model_dir=DEFAULT_MICROBIOME_MODEL_DIR,
                                config=None, progress=None, force=False):
    config = config or MicrobiomeConfig()
    file_path = Path(file_path)
    source_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    existing = model_info(model_dir)
    if (not force and existing and existing.get("task") == "microbiome"
            and existing.get("source_sha256") == source_hash
            and existing.get("config") == asdict(config)
            and existing.get("pipeline_version") == MICROBIOME_PIPELINE_VERSION):
        if progress:
            progress("Reusing saved microbiome model; data and settings are unchanged.")
        return existing
    started = time.perf_counter()
    X, Y = loadandpreprocess(file_path, task="microbiome")
    bundle, metrics, history = train_microbiome(X, Y, config, progress)
    info = {"task": "microbiome", "pipeline_version": MICROBIOME_PIPELINE_VERSION,
            "source": file_path.name, "source_sha256": source_hash,
            "config": asdict(config), "metrics": metrics, "history": history,
            "samples": len(X), "targets": Y.shape[1], "device": "CPU (scikit-learn)",
            "output_units": "Relative abundance (%) and bounded original abundance units",
            "target_ranges": {name: [float(Y[name].min()), float(Y[name].max())] for name in Y},
            "feature_ranges": {name: [float(X[name].min()), float(X[name].max())] for name in X},
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_seconds": round(time.perf_counter() - started, 3)}
    _save_trained_artifacts(bundle, info, model_dir)
    return info


def predict_microbiome_values_controller(user_input_row, model_dir=DEFAULT_MICROBIOME_MODEL_DIR, *, version=None):
    directory = _artifact_directory(model_dir, version)
    if directory is None:
        raise ValueError("No saved microbiome model is available.")
    return _load_bundle(str(directory.resolve())).predict_values(user_input_row)
