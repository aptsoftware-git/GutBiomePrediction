"""Dual prediction UI regressions with stubbed model operations."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class UIFlowTests(unittest.TestCase):
    def setUp(self):
        self.main = ModuleType("main")
        for name, value in {
            "DEFAULT_DATA_PATH": "metabolites.xlsx", "MICROBIOME_DATA_PATH": "microbiome.xlsx",
            "DEFAULT_MODEL_DIR": "default", "CUSTOM_MODEL_DIR": "custom",
            "DEFAULT_MICROBIOME_MODEL_DIR": "microbiome/default",
            "CUSTOM_MICROBIOME_MODEL_DIR": "microbiome/custom",
        }.items():
            setattr(self.main, name, Path(value))
        self.main.TrainingConfig = Mock()
        self.main.MicrobiomeConfig = Mock()
        self.info = {"samples": 39, "targets": 2, "metrics": {"r2": None},
                     "target_ranges": {"A": [139, 292651], "B": [0, 10]}}
        self.main.model_info = Mock(return_value=self.info)
        self.main.list_models = Mock(return_value=[])
        self.main.train_controller = Mock(return_value=self.info)
        self.main.train_microbiome_controller = Mock(return_value=self.info)
        def predict(rows, model_dir, version):
            return pd.DataFrame({"A": [25.5], "B": [74.5]}) if "microbiome" in str(model_dir) else pd.DataFrame({"A": [1], "B": [0]})
        self.main.predict_controller = Mock(side_effect=predict)
        self.main.predict_microbiome_values_controller = Mock(return_value=pd.DataFrame({"A": [2500.5], "B": [4.25]}))
        runtime = ModuleType("runtime")
        runtime.TRAINING_DEVICE = "/CPU:0"
        runtime.DEVICE_NAME = "CPU"
        self.modules = patch.dict(sys.modules, main=self.main, runtime=runtime)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/ui.py"))

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def results(self, task):
        return self.app.session_state["last_dual_prediction"]["outputs"][task]["results"]

    def test_first_run_builds_both_from_separate_files(self):
        self.main.model_info.return_value = None
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.button("Predict").disabled)
        self.button("Build / reuse both models").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.button("Predict").disabled)
        for controller, source, directory in (
            (self.main.train_controller, self.main.DEFAULT_DATA_PATH, self.main.DEFAULT_MODEL_DIR),
            (self.main.train_microbiome_controller, self.main.MICROBIOME_DATA_PATH, self.main.DEFAULT_MICROBIOME_MODEL_DIR),
        ):
            self.assertEqual(controller.call_args.args, (source,))
            self.assertEqual(controller.call_args.kwargs["model_dir"], directory)

    def test_selects_independent_builds_predicts_and_downloads_both(self):
        def builds(directory):
            prefix = "microbiome" if "microbiome" in str(directory) else "metabolite"
            return [{"version": f"run-{prefix}-{i}", "info": dict(self.info, trained_at=f"2026-09-{i:02d}")}
                    for i in range(6, 0, -1)]
        self.main.list_models.side_effect = builds
        self.app.run()
        next(x for x in self.app.selectbox if x.label == "Metabolite saved model build").set_value("run-metabolite-1").run()
        next(x for x in self.app.selectbox if x.label == "Microbiome saved model build").set_value("run-microbiome-2").run()
        self.button("Predict").click().run()
        self.assertFalse(self.app.exception)
        calls = self.main.predict_controller.call_args_list
        self.assertEqual([call.kwargs["version"] for call in calls], ["run-metabolite-1", "run-microbiome-2"])
        pd.testing.assert_frame_equal(calls[0].args[0], calls[1].args[0])
        self.assertEqual(self.results("microbiome")["Relative abundance (%)"].tolist(), [25.5, 74.5])
        self.assertEqual(self.results("metabolite")["Prediction"].tolist(), [1, 0])
        self.assertEqual(self.results("microbiome")["Predicted value"].tolist(), [2500.5, 4.25])
        self.assertNotIn("Training minimum", self.results("microbiome"))
        self.assertNotIn("Training maximum", self.results("microbiome"))
        self.assertIn("Weight (Kgs)", calls[0].args[0])
        self.assertEqual(self.main.predict_microbiome_values_controller.call_args.kwargs["version"], "run-microbiome-2")
        self.assertEqual(len(self.app.get("download_button")), 2)
        html = "".join(item.proto.body for item in self.app.get("html"))
        self.assertIn('aria-label="Microbiome predictions"', html)
        self.assertIn('aria-label="Metabolite predictions"', html)
        self.assertIn("25.5", html)

    def test_pair_survives_reruns_source_changes_and_partial_prediction_failure(self):
        self.app.run()
        self.app.text_input[0].set_value("Original participant")
        self.button("Predict").click().run()
        expected = {task: self.results(task).copy() for task in ("metabolite", "microbiome")}
        self.app.run()
        self.app.radio[0].set_value("Uploaded-data models").run()
        self.main.predict_controller.side_effect = [pd.DataFrame({"new": [0]}), ValueError("test failure")]
        self.app.text_input[0].set_value("New participant")
        self.button("Predict").click().run()
        self.assertFalse(self.app.exception)
        self.assertIn("test failure", self.app.error[0].value)
        for task, result in expected.items():
            pd.testing.assert_frame_equal(self.results(task), result)
        self.assertEqual(self.app.dataframe[0].value.iloc[0]["Name"], "Original participant")

    def test_older_build_keeps_percentage_predictions_and_requests_rebuild(self):
        self.info.pop("target_ranges")
        self.app.run()
        self.button("Predict").click().run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("Predicted value", self.results("microbiome"))
        self.main.predict_microbiome_values_controller.assert_not_called()
        self.assertTrue(any("older build" in item.value for item in self.app.info))

    def test_microbiome_validation_report_is_visible_and_downloadable(self):
        self.info["metrics"]["numeric_values"] = {"per_genus": [
            {"Genus": "Bacteroides", "RMSE": 81047.8, "MAE": 33948.6, "R²": -0.17,
             "Evaluation samples": 39, "Nonzero samples": 39,
             "Baseline RMSE": 82000., "MSE improvement (%)": 2.3}]}
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.dataframe[0].value.iloc[0]["Genus"], "Bacteroides")
        labels = [item.proto.label for item in self.app.get("download_button")]
        self.assertIn("Download microbiome validation report", labels)

    def test_missing_model_and_failed_build_do_not_enable_predict(self):
        self.main.model_info.return_value = None
        self.main.train_microbiome_controller.side_effect = ValueError("bad abundance")
        self.app.run()
        self.button("Build / reuse both models").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.button("Predict").disabled)
        self.assertIn("bad abundance", self.app.error[0].value)

    def test_uploaded_models_require_two_files_and_support_individual_builds(self):
        self.app.run()
        self.app.radio[0].set_value("Uploaded-data models").run()
        self.button("Build / reuse both models").click().run()
        self.main.train_controller.assert_not_called()
        self.main.train_microbiome_controller.assert_not_called()
        self.assertIn("Upload a training file", self.app.error[0].value)
        uploads = [SimpleNamespace(name="metabolites.xlsx", getvalue=lambda: b"metabolites"),
                   SimpleNamespace(name="microbiome.xlsx", getvalue=lambda: b"microbiome")]
        received = []
        def train(path, **kwargs):
            received.append((Path(path).read_bytes(), kwargs["model_dir"]))
            return self.info
        self.main.train_controller.side_effect = train
        self.main.train_microbiome_controller.side_effect = train
        with patch("streamlit.file_uploader", side_effect=uploads):
            self.button("Build / reuse both models").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(received, [(b"metabolites", self.main.CUSTOM_MODEL_DIR),
                                    (b"microbiome", self.main.CUSTOM_MICROBIOME_MODEL_DIR)])
        self.assertFalse(Path(self.main.train_controller.call_args.args[0]).exists())
        with patch("streamlit.file_uploader", side_effect=[None, uploads[1]]):
            self.button("Build / reuse Microbiome model").click().run()
        self.assertEqual(self.main.train_controller.call_count, 1)
        self.assertEqual(self.main.train_microbiome_controller.call_count, 2)


if __name__ == "__main__":
    unittest.main()
