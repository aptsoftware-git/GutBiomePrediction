"""Evaluate reversible per-genus 1–10 coding without changing saved models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from benchmark_microbiome import summarize
from loadandpreprocess import MICROBIOME_DATA_PATH, loadandpreprocess

CANDIDATES = [("mean", 0)] + [("ridge", alpha) for alpha in (1, 10, 100, 1000, 10000)] + [("extra", 3)]


def fit_predict(x, y, train, test, method, parameter):
    # Both mappings use only training rows, including in inner validation.
    features = StandardScaler().fit(x[train])
    targets = MinMaxScaler(feature_range=(1, 10)).fit(y[train])
    coded = targets.transform(y[train])
    if method == "mean":
        predicted = np.tile(coded.mean(axis=0), (len(test), 1))
    else:
        model = (Ridge(alpha=parameter, solver="svd") if method == "ridge" else
                 ExtraTreesRegressor(n_estimators=64, max_depth=parameter, min_samples_leaf=2,
                                     random_state=42, n_jobs=1, max_features=.8))
        model.fit(features.transform(x[train]), coded)
        predicted = model.predict(features.transform(x[test]))
    decoded = targets.inverse_transform(np.clip(predicted, 1, 10))
    # Constant targets must remain constant, including when every observation is zero.
    decoded = np.clip(decoded, targets.data_min_, targets.data_max_)
    return decoded, targets


def benchmark(path):
    X, Y = loadandpreprocess(path, task="microbiome")
    x, y = X.to_numpy(), Y.to_numpy()
    results = {}
    for seed in (42, 7, 123):
        prediction = np.zeros_like(y)
        selected = []
        for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(x), 1):
            scores = []
            for method, parameter in CANDIDATES:
                loss = 0.
                for a, b in KFold(4, shuffle=True, random_state=seed).split(train):
                    pred, mapping = fit_predict(x, y, train[a], train[b], method, parameter)
                    loss += np.sum((mapping.transform(y[train[b]]) - mapping.transform(pred)) ** 2)
                scores.append(loss)
            winner = CANDIDATES[int(np.argmin(scores))]
            selected.append(winner)
            prediction[test], _ = fit_predict(x, y, train, test, *winner)
            print(f"Seed {seed}, fold {fold}: {winner}", flush=True)
        results[seed] = {"metrics": summarize(y, prediction, Y.columns), "selected": selected}
    # Verify pure linear scaling is equivalent to original-target ridge at fixed alpha.
    train, test = next(KFold(5, shuffle=True, random_state=42).split(x))
    scaled, _ = fit_predict(x, y, train, test, "ridge", 100)
    features = StandardScaler().fit(x[train])
    raw = Ridge(alpha=100, solver="svd").fit(features.transform(x[train]), y[train])
    raw_pred = np.clip(raw.predict(features.transform(x[test])), y[train].min(axis=0), y[train].max(axis=0))
    return {"source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "samples": len(X), "targets": len(Y.columns), "feature_range": [1, 10],
            "fixed_alpha_ridge_max_difference": float(np.max(np.abs(scaled - raw_pred))), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(MICROBIOME_DATA_PATH))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(benchmark(args.data), indent=2))
