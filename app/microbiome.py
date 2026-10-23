"""Small-sample, multi-output abundance regression with nested model selection."""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler

from loadandpreprocess import prepare_features


@dataclass(frozen=True)
class MicrobiomeConfig:
    validation: str = "kfold"
    output_mode: str = "relative"
    seed: int = 42

    def __post_init__(self):
        if self.validation not in {"holdout", "kfold", "loo", "none"}:
            raise ValueError("Unknown validation method.")
        if self.output_mode not in {"relative", "original"}:
            raise ValueError("Unknown microbiome output mode.")


def composition(values):
    values = np.asarray(values, dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Each microbiome training row must have a positive total abundance.")
    return values / totals


def transform_targets(values, method):
    if method == "log1p":
        return np.log1p(values)
    proportions = composition(values)
    if method == "hellinger":
        return np.sqrt(proportions)
    # Fixed relative pseudocount: independent of sequencing depth and test rows.
    logs = np.log(proportions + 1e-6)
    return logs - logs.mean(axis=1, keepdims=True)


@dataclass
class MicrobiomeBundle:
    regressor: object
    x_scaler: object
    feature_columns: list
    target_columns: list
    output_mode: str
    method: str
    alpha: object
    baseline: np.ndarray
    observed: np.ndarray
    training_min: np.ndarray
    training_max: np.ndarray
    target_min: object = None
    target_max: object = None
    value_model: object = None

    def predict_values(self, frame):
        if self.value_model is None:
            raise ValueError("This older microbiome build only predicts percentages. Rebuild it to add numeric values and ranges.")
        return self.value_model.predict(frame)

    def predict(self, frame):
        features = prepare_features(frame, self.feature_columns)
        if self.regressor is None:
            values = np.tile(self.baseline, (len(features), 1))
        else:
            transformed = self.regressor.predict(self.x_scaler.transform(features))
            if self.method == "log1p":
                transformed = np.clip(transformed, np.log1p(self.target_min), np.log1p(self.target_max))
                values = np.expm1(transformed)
            elif self.method == "hellinger":
                values = np.maximum(transformed, 0) ** 2
            else:
                values = np.exp(transformed - transformed.max(axis=1, keepdims=True))
            values[:, ~self.observed] = 0
            if self.output_mode == "relative":
                empty = values.sum(axis=1) <= 0
                values[empty] = self.baseline
                values = composition(values)
        if self.output_mode == "original":
            values = np.clip(values, self.target_min, self.target_max)
        else:
            values = values * 100
        return pd.DataFrame(values, columns=self.target_columns, index=frame.index)


def fit_candidate(X, Y, config, method="mean", alpha=None):
    values = Y.to_numpy(dtype=float)
    # Hellinger centroid is the constant profile minimizing our selection loss.
    # Using an arithmetic mean here would unfairly favor the transformed regressors.
    baseline = (composition(np.sqrt(composition(values)).mean(axis=0, keepdims=True) ** 2)[0]
                if config.output_mode == "relative" else np.expm1(np.log1p(values).mean(axis=0)))
    scaler = StandardScaler().fit(X)
    regressor = None
    if method != "mean":
        regressor = Ridge(alpha=alpha, solver="svd")
        regressor.fit(scaler.transform(X), transform_targets(values, method))
    return MicrobiomeBundle(regressor, scaler, list(X.columns), list(Y.columns),
                            config.output_mode, method, alpha, baseline,
                            (values > 0).any(axis=0), X.min().to_numpy(), X.max().to_numpy(),
                            values.min(axis=0), values.max(axis=0))


def selection_loss(truth, prediction, mode="relative"):
    if mode == "original":
        return float(np.mean((np.log1p(truth) - np.log1p(prediction)) ** 2))
    # Mean squared Hellinger distance: zeros are valid, no fitted pseudocount.
    return float(np.mean(np.sum((np.sqrt(composition(truth)) -
                                  np.sqrt(prediction / 100)) ** 2, axis=1) / 2))


def select_and_fit(X, Y, config):
    methods = ("hellinger", "clr") if config.output_mode == "relative" else ("log1p",)
    candidates = [("mean", None)] + [(method, alpha) for method in methods
                                    for alpha in (1., 10., 100., 1000., 10000.)]
    splits = list(KFold(n_splits=min(4, len(X)), shuffle=True,
                        random_state=config.seed).split(X))
    scores = []
    for method, alpha in candidates:
        total, count = 0., 0
        for train, test in splits:
            candidate = fit_candidate(X.iloc[train], Y.iloc[train], config, method, alpha)
            prediction = candidate.predict(X.iloc[test]).to_numpy()
            total += len(test) * selection_loss(Y.iloc[test].to_numpy(), prediction, config.output_mode)
            count += len(test)
        scores.append({"method": method, "alpha": alpha, "validation_loss": total / count})
    # Ties prefer the simpler population baseline, which is listed first.
    selected = min(scores, key=lambda item: item["validation_loss"])
    bundle = fit_candidate(X, Y, config, selected["method"], selected["alpha"])
    return bundle, scores


def _train_profile(X, Y, config, progress=None):
    if len(X) < 6:
        raise ValueError("Microbiome training requires at least 6 independent participants.")
    indexes = np.arange(len(X))
    if config.validation == "holdout":
        splits = [train_test_split(indexes, test_size=max(2, int(np.ceil(len(X) * .2))),
                                   random_state=config.seed)]
    elif config.validation == "kfold":
        splits = list(KFold(n_splits=min(5, len(X) // 2), shuffle=True,
                            random_state=config.seed).split(X))
    elif config.validation == "loo":
        splits = list(LeaveOneOut().split(X))
    else:
        splits = []
    actual, predictions, baselines, folds = [], [], [], []
    for number, (train, test) in enumerate(splits, 1):
        if progress:
            progress(f"Microbiome {config.output_mode} evaluation {number}/{len(splits)}: selecting on training rows only")
        bundle, _ = select_and_fit(X.iloc[train], Y.iloc[train], config)
        predicted = bundle.predict(X.iloc[test]).to_numpy()
        baseline = fit_candidate(X.iloc[train], Y.iloc[train], config).predict(X.iloc[test]).to_numpy()
        truth = Y.iloc[test].to_numpy()
        loss = selection_loss(truth, predicted, config.output_mode)
        base_loss = selection_loss(truth, baseline, config.output_mode)
        folds.append({"fold": number, "samples": len(test), "method": bundle.method,
                      "alpha": bundle.alpha, "loss": loss, "baseline_loss": base_loss})
        actual.append(composition(truth) * 100 if config.output_mode == "relative" else truth)
        predictions.append(predicted)
        baselines.append(baseline)
    metrics = {"validation": config.validation, "evaluation_samples": 0,
               "r2": None, "rmse": None, "mae": None, "selection_loss": None,
               "baseline_loss": None, "skill_vs_baseline": None, "folds": folds,
               "loss_name": "squared Hellinger distance" if config.output_mode == "relative" else "log1p MSE"}
    if actual:
        truth, predicted, baseline = map(np.concatenate, (actual, predictions, baselines))
        loss = selection_loss(truth, predicted, config.output_mode)
        base_loss = selection_loss(truth, baseline, config.output_mode)
        variable = np.ptp(truth, axis=0) > 0
        metrics.update(evaluation_samples=len(truth),
                       r2=float(r2_score(truth[:, variable], predicted[:, variable])) if variable.any() else None,
                       rmse=float(np.sqrt(mean_squared_error(truth, predicted))),
                       mae=float(mean_absolute_error(truth, predicted)),
                       selection_loss=loss, baseline_loss=base_loss,
                       skill_vs_baseline=float(1 - loss / base_loss) if base_loss > 0 else None,
                       variable_targets=int(variable.sum()))
    if actual and config.output_mode == "original":
        errors = np.mean((truth - predicted) ** 2, axis=0)
        baseline_errors = np.mean((truth - baseline) ** 2, axis=0)
        variances = np.mean((truth - truth.mean(axis=0)) ** 2, axis=0)
        absolute_errors = np.mean(np.abs(truth - predicted), axis=0)
        metrics["per_genus"] = [
            {"Genus": name, "Evaluation samples": len(truth),
             "Nonzero samples": int(np.count_nonzero(truth[:, index])),
             "RMSE": float(np.sqrt(errors[index])), "MAE": float(absolute_errors[index]),
             "R²": float(1 - errors[index] / variances[index]) if variances[index] > 0 else None,
             "Baseline RMSE": float(np.sqrt(baseline_errors[index])),
             "MSE improvement (%)": float(100 * (1 - errors[index] / baseline_errors[index])) if baseline_errors[index] > 0 else None}
            for index, name in enumerate(Y.columns)
        ]
    if progress:
        progress(f"Selecting and fitting the saved microbiome model on all {len(X)} participants")
    bundle, candidates = select_and_fit(X, Y, config)
    history = {"selected_method": bundle.method, "selected_alpha": bundle.alpha,
               "inner_cv_candidates": candidates,
               "never_observed_targets": int((~bundle.observed).sum())}
    return bundle, metrics, history


def train_microbiome(X, Y, config, progress=None):
    """Fit independently validated percentage and bounded original-value predictors."""
    bundle, metrics, history = _train_profile(X, Y, replace(config, output_mode="relative"), progress)
    values, value_metrics, value_history = _train_profile(X, Y, replace(config, output_mode="original"), progress)
    bundle.value_model = values
    metrics["numeric_values"] = value_metrics
    history["numeric_values"] = value_history
    return bundle, metrics, history
