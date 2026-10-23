"""Abundance semantics, nested validation, persistence and input validation."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from loadandpreprocess import (FEATURE_COLUMNS, MICROBIOME_DATA_PATH,
                               loadandpreprocess, prepare_features)
from microbiome import (MicrobiomeConfig, composition, fit_candidate, select_and_fit,
                         train_microbiome)
from main import (train_microbiome_controller, predict_controller, predict_microbiome_values_controller, list_models,
                  _artifact_directory, _load_bundle)


class MicrobiomeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.normal(size=(12, 3)), columns=["Age", "BMI", "Sex"])
        self.X["Sex"] = [0, 1] * 6
        self.Y = pd.DataFrame(rng.poisson(4, size=(12, 4)), columns=list("ABCD"))
        self.Y["unobserved"] = 0
        self.config = MicrobiomeConfig()

    def test_each_candidate_is_nonnegative_closed_and_retains_zero_targets(self):
        for method in ("mean", "hellinger", "clr"):
            bundle = fit_candidate(self.X, self.Y, self.config, method, 100)
            result = bundle.predict(self.X.iloc[:2])
            self.assertTrue(np.isfinite(result.to_numpy()).all())
            self.assertTrue((result.to_numpy() >= 0).all())
            np.testing.assert_allclose(result.sum(axis=1), 100)
            np.testing.assert_array_equal(result["unobserved"], 0)
            pd.testing.assert_frame_equal(result, bundle.predict(self.X.iloc[:2, ::-1]))
            self.assertTrue(((result.to_numpy() > 0) & (result.to_numpy() < 100)).any())

    def test_numeric_predictions_respect_per_genus_bounds_even_for_extreme_inputs(self):
        y = self.Y.copy()
        y["constant"] = 17.5
        y["positive_min"] = np.arange(len(y)) + 139
        for method in ("mean", "log1p"):
            bundle = fit_candidate(self.X, y, MicrobiomeConfig(output_mode="original"), method, 1.)
            extreme = self.X.iloc[:2].copy()
            extreme["Age"] = [-1e10, 1e10]
            prediction = bundle.predict(extreme)
            self.assertTrue(np.isfinite(prediction).all().all())
            self.assertTrue((prediction >= y.min()).all().all())
            self.assertTrue((prediction <= y.max()).all().all())
            np.testing.assert_array_equal(prediction["constant"], 17.5)
            np.testing.assert_array_equal(prediction["unobserved"], 0)

    def test_benchmark_candidates_use_training_bounds_only(self):
        from benchmark_microbiome import RAW_CANDIDATES, SCREENING_CANDIDATES, candidate_prediction
        x, y = self.X.to_numpy(), self.Y.to_numpy(dtype=float)
        train, heldout = np.arange(8), np.arange(8, 12)
        original = y[train].copy()
        y[heldout] = 1e8  # Test labels must not change either the fit or the bounds.
        for candidate in RAW_CANDIDATES + SCREENING_CANDIDATES:
            prediction = candidate_prediction(x, y, train, heldout, candidate)
            self.assertEqual(prediction.shape, (4, y.shape[1]))
            self.assertTrue((prediction >= original.min(axis=0)).all())
            self.assertTrue((prediction <= original.max(axis=0)).all())
            np.testing.assert_allclose(prediction, candidate_prediction(
                x, self.Y.to_numpy(dtype=float), train, heldout, candidate))

    def test_minmax_coding_roundtrip_and_fold_local_bounds(self):
        from benchmark_minmax import fit_predict
        x, y = self.X.to_numpy(), self.Y.to_numpy(dtype=float)
        train, heldout = np.arange(8), np.arange(8, 12)
        for method, parameter in (("mean", 0), ("ridge", 100), ("extra", 3)):
            predicted, mapping = fit_predict(x, y, train, heldout, method, parameter)
            np.testing.assert_allclose(mapping.inverse_transform(mapping.transform(y[train])), y[train], atol=1e-12)
            np.testing.assert_allclose(mapping.transform(mapping.data_min_.reshape(1, -1)), 1)
            self.assertTrue((predicted >= y[train].min(axis=0)).all())
            self.assertTrue((predicted <= y[train].max(axis=0)).all())
            np.testing.assert_array_equal(predicted[:, -1], 0)
            changed = y.copy()
            changed[heldout] = 1e8
            other, _ = fit_predict(x, changed, train, heldout, method, parameter)
            np.testing.assert_allclose(predicted, other)

    def test_weight_header_is_canonical_and_legacy_saved_schema_still_works(self):
        for source in ("Weight (cm)", "Weight (kg)", "Weight (Kgs)"):
            frame = pd.DataFrame({source: [72.25]})
            self.assertEqual(list(prepare_features(frame)), ["Weight (Kgs)"])
            for saved in ("Weight (cm)", "Weight (Kgs)"):
                actual = prepare_features(frame, [saved])
                self.assertEqual(list(actual), [saved])
                self.assertEqual(actual.iloc[0, 0], 72.25)
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_features(pd.DataFrame({"Weight (cm)": [70], "Weight (Kgs)": [71]}))

    def test_sequencing_depth_does_not_change_relative_model(self):
        scaled = self.Y.mul(np.arange(1, len(self.Y) + 1), axis=0)
        for method in ("mean", "hellinger", "clr"):
            first = fit_candidate(self.X, self.Y, self.config, method, 100).predict(self.X)
            second = fit_candidate(self.X, scaled, self.config, method, 100).predict(self.X)
            np.testing.assert_allclose(first, second, atol=1e-9)

    def test_all_outer_folds_select_only_on_their_training_rows(self):
        outer_training_sets = []
        real_select = select_and_fit
        def select(x, y, config):
            outer_training_sets.append(set(x.index))
            allowed = set(x.index)
            def fit(inner_x, inner_y, *args, **kwargs):
                self.assertTrue(set(inner_x.index).issubset(allowed))
                pd.testing.assert_index_equal(inner_x.index, inner_y.index)
                candidate = fit_candidate(inner_x, inner_y, *args, **kwargs)
                self.assertEqual(candidate.x_scaler.n_samples_seen_, len(inner_x))
                np.testing.assert_array_equal(candidate.target_min, inner_y.min().to_numpy())
                np.testing.assert_array_equal(candidate.target_max, inner_y.max().to_numpy())
                return candidate
            with patch("microbiome.fit_candidate", side_effect=fit):
                return real_select(x, y, config)
        with patch("microbiome.select_and_fit", side_effect=select):
            _, metrics, _ = train_microbiome(self.X, self.Y, self.config)
        self.assertEqual(metrics["evaluation_samples"], 12)
        self.assertEqual(len(outer_training_sets), 12)
        for groups in (outer_training_sets[:6], outer_training_sets[6:]):
            self.assertTrue(all(len(rows) < 12 for rows in groups[:-1]))
            self.assertEqual(groups[-1], set(self.X.index))
            heldout = [set(self.X.index) - rows for rows in groups[:-1]]
            self.assertEqual(sorted(index for rows in heldout for index in rows), list(range(12)))
        self.assertIsNotNone(metrics["baseline_loss"])
        report = metrics["numeric_values"]["per_genus"]
        self.assertEqual([item["Genus"] for item in report], list(self.Y))
        self.assertTrue(all(item["Evaluation samples"] == 12 for item in report))
        zero_target = next(item for item in report if item["Genus"] == "unobserved")
        self.assertEqual(zero_target["RMSE"], 0)
        self.assertIsNone(zero_target["R²"])
        self.assertIsNone(zero_target["MSE improvement (%)"])
        for item in report:
            self.assertGreaterEqual(item["RMSE"] + 1e-12, item["MAE"])

    def test_train_only_does_not_claim_independent_metrics(self):
        _, metrics, history = train_microbiome(self.X, self.Y, MicrobiomeConfig(validation="none"))
        self.assertIsNone(metrics["rmse"])
        self.assertIsNone(metrics["skill_vs_baseline"])
        self.assertEqual(metrics["evaluation_samples"], 0)
        self.assertNotIn("per_genus", metrics["numeric_values"])
        self.assertEqual(len(history["inner_cv_candidates"]), 11)

    def test_population_baseline_wins_for_constant_profiles(self):
        y = pd.DataFrame(np.tile([1, 3, 0], (12, 1)), columns=list("ABC"))
        bundle, _ = select_and_fit(self.X, y, self.config)
        self.assertEqual(bundle.method, "mean")
        np.testing.assert_allclose(bundle.predict(self.X), np.tile([25, 75, 0], (12, 1)))

    def test_file_preserves_numeric_targets_and_rejects_bad_rows(self):
        X, Y = loadandpreprocess(MICROBIOME_DATA_PATH, task="microbiome")
        self.assertEqual(X.shape, (39, 10))
        self.assertEqual(Y.shape, (39, 1638))
        self.assertEqual(Y.iloc[0, 0], 292651)
        self.assertEqual(list(X), FEATURE_COLUMNS)
        frame = pd.read_excel(MICROBIOME_DATA_PATH).iloc[:8, :15]
        with TemporaryDirectory() as root:
            path = Path(root) / "input.xlsx"
            for mutation, pattern in (
                (lambda df: df.iloc.__setitem__((0, 11), -1), "nonnegative"),
                (lambda df: df.iloc.__setitem__((0, slice(11, None)), 0), "positive total"),
                (lambda df: df.iloc.__setitem__((0, 11), np.nan), "missing"),
                (lambda df: df.iloc.__setitem__((1, 0), df.iloc[0, 0]), "unique"),
            ):
                invalid = frame.copy()
                mutation(invalid)
                invalid.to_excel(path, index=False)
                with self.assertRaisesRegex(ValueError, pattern):
                    loadandpreprocess(path, task="microbiome")
            frame.rename(columns={"Weight (cm)": "Weight (Kgs)"}).to_excel(path, index=False)
            features, _ = loadandpreprocess(path, task="microbiome")
            self.assertIn("Weight (Kgs)", features)
        with self.assertRaisesRegex(ValueError, "at least 6"):
            train_microbiome(self.X.iloc[:5], self.Y.iloc[:5], self.config)

    def test_save_reload_reuse_versions_and_failure_preserves_model(self):
        frame = pd.read_excel(MICROBIOME_DATA_PATH).iloc[:10, :16]
        with TemporaryDirectory() as root:
            path = Path(root) / "input.xlsx"
            directory = Path(root) / "models"
            frame.to_excel(path, index=False)
            config = MicrobiomeConfig(validation="none")
            info = train_microbiome_controller(path, model_dir=directory, config=config)
            first_version = _artifact_directory(directory).name
            rows = frame.iloc[:2, 1:11]
            prediction = predict_controller(rows, directory)
            numeric = predict_microbiome_values_controller(rows, directory)
            self.assertTrue((numeric.to_numpy() >= frame.iloc[:, 11:].min().to_numpy()).all())
            self.assertTrue((numeric.to_numpy() <= frame.iloc[:, 11:].max().to_numpy()).all())
            self.assertEqual(info["target_ranges"][frame.columns[11]], [float(frame.iloc[:, 11].min()), float(frame.iloc[:, 11].max())])
            self.assertEqual(prediction.shape, (2, 5))
            np.testing.assert_allclose(prediction.sum(axis=1), 100)
            _load_bundle.cache_clear()
            pd.testing.assert_frame_equal(prediction, predict_controller(rows, directory))
            pd.testing.assert_frame_equal(numeric, predict_microbiome_values_controller(rows, directory))
            with patch("main.train_microbiome", side_effect=AssertionError("should reuse")):
                self.assertEqual(info, train_microbiome_controller(path, model_dir=directory, config=config))
            with patch("main.train_microbiome", side_effect=ValueError("fit failure")):
                with self.assertRaisesRegex(ValueError, "fit failure"):
                    train_microbiome_controller(path, model_dir=directory, config=config, force=True)
            self.assertEqual(_artifact_directory(directory).name, first_version)
            with patch("main.pickle.dump", side_effect=ValueError("save failure")):
                with self.assertRaisesRegex(ValueError, "save failure"):
                    train_microbiome_controller(path, model_dir=directory, config=config, force=True)
            self.assertEqual(_artifact_directory(directory).name, first_version)
            self.assertFalse(list(directory.glob(".training-*")))
            train_microbiome_controller(path, model_dir=directory, config=config, force=True)
            self.assertEqual(len(list_models(directory)), 2)
            pd.testing.assert_frame_equal(prediction, predict_controller(rows, directory, version=first_version))
            _load_bundle.cache_clear()


if __name__ == "__main__":
    unittest.main()
