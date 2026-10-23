"""Compare microbiome alternatives without changing any saved prediction model.

Run: OPENBLAS_NUM_THREADS=1 python app/benchmark_microbiome.py --output report.json
The optional BLAS limit avoids overhead on the small matrices in this dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from loadandpreprocess import MICROBIOME_DATA_PATH, loadandpreprocess
from microbiome import MicrobiomeConfig, composition, select_and_fit

RAW_CANDIDATES = [("mean", 0), ("ridge", 10), ("ridge", 100), ("ridge", 1000),
                  ("pls", 1), ("pls", 2), ("extra", 3), ("extra", 6), ("forest", 3)]
SCREENING_CANDIDATES = [("knn", 3), ("knn", 5), ("kernel", 1), ("kernel", 10)]


def candidate_prediction(x, y, train, test, candidate, mode="original"):
    """All preprocessing and bounds are fitted exclusively on train indexes."""
    name, parameter = candidate
    scaler = StandardScaler().fit(x[train])
    features = scaler.transform(x[train])
    heldout = scaler.transform(x[test])
    targets = y[train] if mode == "original" else np.sqrt(composition(y[train]))
    if name == "mean":
        predicted = np.tile(targets.mean(axis=0), (len(test), 1))
    else:
        if name == "ridge":
            model = Ridge(alpha=parameter, solver="svd")
        elif name == "pls":
            model = PLSRegression(n_components=min(parameter, features.shape[1], len(train) - 1), scale=False)
        elif name in {"extra", "forest"}:
            estimator = ExtraTreesRegressor if name == "extra" else RandomForestRegressor
            model = estimator(n_estimators=64, max_depth=parameter, min_samples_leaf=2,
                              random_state=42, n_jobs=1, max_features=.8)
        elif name == "knn":
            model = KNeighborsRegressor(n_neighbors=min(parameter, len(train)), weights="distance")
        elif name == "kernel":
            model = KernelRidge(kernel="rbf", alpha=parameter, gamma=.1)
        else:
            raise ValueError(f"Unknown benchmark candidate: {name}")
        model.fit(features, targets)
        predicted = model.predict(heldout)
    if mode == "original":
        return np.clip(predicted, y[train].min(axis=0), y[train].max(axis=0))
    return composition(np.maximum(predicted, 0) ** 2) * 100


def summarize(truth, predicted, columns):
    result = {"rmse": float(np.sqrt(np.mean((truth - predicted) ** 2))),
              "mae": float(np.mean(np.abs(truth - predicted))),
              "log_mse": float(np.mean((np.log1p(truth) - np.log1p(predicted)) ** 2))}
    if "Bacteroides" in columns:
        index = list(columns).index("Bacteroides")
        result["bacteroides_rmse"] = float(np.sqrt(np.mean((truth[:, index] - predicted[:, index]) ** 2)))
        result["bacteroides_mae"] = float(np.mean(np.abs(truth[:, index] - predicted[:, index])))
    return result


def benchmark(path, seeds=(42, 7, 123), progress=print):
    X, Y = loadandpreprocess(path, task="microbiome")
    x, y = X.to_numpy(), Y.to_numpy()
    if len(X) < 10:
        raise ValueError("This benchmark uses five outer and four inner folds and needs at least 10 rows.")
    nested, screening = {}, {}
    for seed in seeds:
        old, new, baseline = (np.zeros_like(y) for _ in range(3))
        selected = []
        splits = list(KFold(5, shuffle=True, random_state=seed).split(x))
        for fold, (train, test) in enumerate(splits, 1):
            losses = []
            for candidate in RAW_CANDIDATES:
                loss = 0.
                for a, b in KFold(4, shuffle=True, random_state=seed).split(train):
                    pred = candidate_prediction(x, y, train[a], train[b], candidate)
                    loss += np.sum((y[train[b]] - pred) ** 2)
                losses.append(loss)
            winner = RAW_CANDIDATES[int(np.argmin(losses))]
            selected.append(winner)
            new[test] = candidate_prediction(x, y, train, test, winner)
            baseline[test] = candidate_prediction(x, y, train, test, ("mean", 0))
            model, _ = select_and_fit(X.iloc[train], Y.iloc[train],
                                     MicrobiomeConfig(output_mode="original", seed=seed))
            old[test] = model.predict(X.iloc[test]).to_numpy()
            progress(f"Seed {seed}, outer fold {fold}/5: {winner}")
        nested[seed] = {"old": summarize(y, old, Y.columns), "new": summarize(y, new, Y.columns),
                        "baseline": summarize(y, baseline, Y.columns), "selected": selected}
        for mode in ("original", "relative"):
            for candidate in SCREENING_CANDIDATES:
                output = np.zeros_like(y)
                for train, test in splits:
                    output[test] = candidate_prediction(x, y, train, test, candidate, mode)
                truth = y if mode == "original" else composition(y) * 100
                scores = summarize(truth, output, Y.columns)
                scores["hellinger"] = float(np.mean(np.sum((np.sqrt(composition(truth)) -
                                            np.sqrt(composition(output))) ** 2, axis=1) / 2))
                key = f"{mode}_{candidate[0]}{candidate[1]}"
                screening.setdefault(key, {})[seed] = scores
    return {"source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "samples": len(X), "targets": Y.shape[1], "seeds": list(seeds),
            "nested_numeric": nested, "fixed_candidate_screening": screening}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(MICROBIOME_DATA_PATH))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = benchmark(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2))
