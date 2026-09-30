"""Run with: .venv/bin/python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from loadandpreprocess import DEFAULT_DATA_PATH, prepare_features
from main import (TrainingConfig, _artifact_directory, _load_bundle, _save_trained_artifacts,
                  list_models, model_training, predict_controller, train_controller)
from runtime import tf
from training import fit_in_memory


class FeatureTests(unittest.TestCase):
    def test_sex_encodes_all_pandas_string_dtypes(self):
        for dtype in (object, "str", "string", "category"):
            with self.subTest(dtype=dtype):
                frame = pd.DataFrame({"Sex": pd.Series(["Male", " female "], dtype=dtype)})
                self.assertEqual(prepare_features(frame)["Sex"].tolist(), [1, 0])

    def test_numeric_sex_is_supported(self):
        frame = pd.DataFrame({"Sex": [1, 0, 1.0, 0.0]})
        self.assertEqual(prepare_features(frame)["Sex"].tolist(), [1, 0, 1, 0])

    def test_saved_feature_order_and_extra_name(self):
        frame = pd.DataFrame({"Sex": ["Male"], "Name": ["test"], "Age": [30]})
        actual = prepare_features(frame, ["Age", "Sex"])
        self.assertEqual(list(actual.columns), ["Age", "Sex"])
        np.testing.assert_array_equal(actual.to_numpy(), [[30, 1]])

    def test_invalid_input_has_clear_error(self):
        for frame, pattern in (
            (pd.DataFrame({"Sex": ["unknown"]}), "Sex must"),
            (pd.DataFrame({"Sex": [None]}), "Sex must"),
            (pd.DataFrame({"Age": ["abc"]}), "Age must"),
            (pd.DataFrame({"Age": [np.inf]}), "infinite"),
            (pd.DataFrame({"Age": [np.nan]}), "missing"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                prepare_features(frame)
        with self.assertRaisesRegex(ValueError, "Missing required features: Age"):
            prepare_features(pd.DataFrame({"Sex": ["Male"]}), ["Age", "Sex"])


class TrainingTests(unittest.TestCase):
    def test_history_keeps_six_builds_and_loads_selected_version(self):
        from types import SimpleNamespace
        network = SimpleNamespace(save=lambda path: Path(path).write_bytes(b"test artifact"))
        bundle = SimpleNamespace(predictor=network, decoder=network, x_scaler=None,
                                 y_scaler=None, feature_columns=[], target_columns=[])
        with TemporaryDirectory() as root:
            for i in range(1, 7):
                _save_trained_artifacts(bundle, {"trained_at": f"2026-09-{i:02d}"}, root)
            history = list_models(root)
            self.assertEqual(len(history), 6)
            self.assertEqual(history[0]["info"]["trained_at"], "2026-09-06")
            oldest = history[-1]["version"]
            current = _artifact_directory(root)
            rows = pd.DataFrame({"Age": [30]})
            with patch("main._load_bundle") as load:
                predict_controller(rows, root, version=oldest)
                load.assert_called_once_with(str((Path(root) / oldest).resolve()))
                load.return_value.predict.assert_called_once_with(rows)
            self.assertEqual(_artifact_directory(root), current)
            incomplete = Path(root) / "run-incomplete"
            incomplete.mkdir()
            (incomplete / "training.json").write_text("{}")
            self.assertEqual(len(list_models(root)), 6)
            with self.assertRaises(ValueError):
                predict_controller(rows, root, version="../outside")

    def test_compiled_loop_learns(self):
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(.05), loss="mse")
        x = np.linspace(-1, 1, 16).reshape(-1, 1).astype("float32")
        y = 2 * x + .3
        before = np.mean((model(x).numpy() - y) ** 2)
        result = fit_in_memory(model, x, y, epochs=150, batch_size=16, patience=30)
        after = np.mean((model(x).numpy() - y) ** 2)
        self.assertLess(after, before * .05)
        self.assertGreater(result["epochs"], 1)

    def test_evaluation_and_final_fit_use_correct_rows(self):
        X = pd.DataFrame({"Age": range(10), "Sex": [0, 1] * 5})
        Y = pd.DataFrame({"target": [0, 1] * 5})
        fits = []
        class FakeBundle:
            def __init__(self, train):
                self.train = set(train)
            def predict(self, rows):
                self_test.assertTrue(self.train.isdisjoint(rows.index))
                return Y.loc[rows.index]
        self_test = self
        def fit(x, y, config, progress):
            fits.append(list(x.index))
            return FakeBundle(x.index), {}
        for method, expected in (("holdout", 2), ("kfold", 6), ("loo", 11), ("none", 1)):
            with self.subTest(method=method), patch("main._fit_models", side_effect=fit):
                fits.clear()
                _, metrics, _ = model_training(X, Y, TrainingConfig(validation=method))
                self.assertEqual(len(fits), expected)
                self.assertEqual(fits[-1], list(X.index))
                self.assertEqual(metrics["evaluation_samples"], {"holdout": 2, "kfold": 10, "loo": 10, "none": 0}[method])
                self.assertEqual(metrics["r2"], None if method == "none" else 1)

    def test_save_reload_reuse_and_failure_preserves_previous_model(self):
        with TemporaryDirectory() as root:
            directory = Path(root) / "model"
            config = TrainingConfig(epochs=2, patience=2, validation="none")
            info = train_controller(DEFAULT_DATA_PATH, model_dir=directory, config=config)
            version = _artifact_directory(directory)
            self.assertEqual(info["samples"], 39)
            rows = pd.read_excel(DEFAULT_DATA_PATH).iloc[:2, 1:11]
            first = predict_controller(rows, directory)
            self.assertEqual(first.shape, (2, 286))
            self.assertTrue(np.isin(first.to_numpy(), [0, 1]).all())
            _load_bundle.cache_clear()
            pd.testing.assert_frame_equal(first, predict_controller(rows[rows.columns[::-1]], directory))
            with patch("main.model_training", side_effect=AssertionError("must reuse")):
                again = train_controller(DEFAULT_DATA_PATH, model_dir=directory, config=config)
            self.assertEqual(info, again)
            with patch("main.model_training", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    train_controller(DEFAULT_DATA_PATH, model_dir=directory, config=config, force=True)
            saved = _load_bundle(str(version.resolve()))
            with patch.object(saved.decoder, "save", side_effect=RuntimeError("save failure")):
                with self.assertRaisesRegex(RuntimeError, "save failure"):
                    _save_trained_artifacts(saved, info, directory)
            self.assertFalse(list(directory.glob(".training-*")))
            self.assertEqual(_artifact_directory(directory), version)
            pd.testing.assert_frame_equal(first, predict_controller(rows, directory))
            self.assertEqual(json.loads((version / "training.json").read_text())["source_sha256"], info["source_sha256"])
            _load_bundle.cache_clear()


if __name__ == "__main__":
    unittest.main()
