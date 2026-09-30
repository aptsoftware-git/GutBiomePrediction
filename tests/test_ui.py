"""UI flow regressions using stubbed model operations (no training required)."""
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class UIFlowTests(unittest.TestCase):
    def setUp(self):
        self.main = ModuleType("main")
        self.main.DEFAULT_DATA_PATH = Path("bundled.xlsx")
        self.main.DEFAULT_MODEL_DIR = Path("default")
        self.main.CUSTOM_MODEL_DIR = Path("custom")
        self.main.TrainingConfig = Mock()
        self.info = {"samples": 39, "targets": 2,
                     "metrics": {"r2": None}}
        self.main.model_info = Mock(return_value=self.info)
        self.main.list_models = Mock(return_value=[])
        self.main.train_controller = Mock(return_value=self.info)
        self.main.predict_controller = Mock(return_value=pd.DataFrame({"A": [1], "B": [0]}))
        runtime = ModuleType("runtime")
        runtime.TRAINING_DEVICE = "/CPU:0"
        runtime.DEVICE_NAME = "CPU"
        self.modules = patch.dict(sys.modules, main=self.main, runtime=runtime)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/ui.py"))

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def test_first_run_build_enables_prediction(self):
        self.main.model_info.return_value = None
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.button("Predict").disabled)
        self.button("Build default model").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.button("Predict").disabled)
        self.assertEqual(self.main.train_controller.call_args.args, (self.main.DEFAULT_DATA_PATH,))
        self.assertEqual(self.main.train_controller.call_args.kwargs["model_dir"], self.main.DEFAULT_MODEL_DIR)

    def test_select_older_build_for_prediction(self):
        self.main.list_models.return_value = [
            {"version": f"run-{i}", "info": dict(self.info, trained_at=f"2026-09-{i:02d}")}
            for i in range(6, 0, -1)
        ]
        self.app.run()
        selector = next(item for item in self.app.selectbox if item.label == "Saved model build")
        selector.set_value("run-1").run()
        self.button("Predict").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.main.predict_controller.call_args.kwargs["version"], "run-1")
        self.assertEqual(self.app.title[0].value, "Microbiome Prediction")
        self.assertFalse(self.app.json)
        self.assertNotIn("Preview", [item.value for item in self.app.subheader])

    def test_result_survives_reruns_and_failed_predictions(self):
        self.app.run()
        self.app.text_input[0].set_value("Original participant")
        self.button("Predict").click().run()
        expected = self.app.dataframe[-1].value.copy()
        self.app.run()  # Same full rerun that used to erase downloaded results.
        pd.testing.assert_frame_equal(self.app.dataframe[-1].value, expected)
        self.main.predict_controller.assert_called_once()
        self.app.radio[0].set_value("Uploaded-data model").run()
        pd.testing.assert_frame_equal(self.app.dataframe[-1].value, expected)
        self.assertIn("Default GCMS model", self.app.caption[-2].value)
        self.main.predict_controller.side_effect = ValueError("test failure")
        self.app.text_input[0].set_value("New participant")
        self.button("Predict").click().run()
        self.assertFalse(self.app.exception)
        pd.testing.assert_frame_equal(self.app.dataframe[-1].value, expected)
        self.assertEqual(self.app.dataframe[0].value.iloc[0]["Name"], "Original participant")


if __name__ == "__main__":
    unittest.main()
