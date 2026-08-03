import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
REPO_DIR = Path(__file__).resolve().parent
LOGOSDATA_DIR = REPO_DIR / "logosdata"
for module_dir in (str(REPO_DIR), str(LOGOSDATA_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from cats_batch import CATS_batch
from logos_data import MainWindow
from logos_instruments_insitu import CATS_Instrument


class FakeButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class CalibrationButtonTests(unittest.TestCase):
    def test_cats_offers_air_and_calibration_run_filters(self):
        self.assertEqual(
            CATS_Instrument.RUN_TYPE_MAP,
            {
                "All": None,
                "Air Samples": "air",
                "Calibrations": "cal",
            },
        )

    def test_cats_cal_chunk_enables_calibration_without_run_type_mapping(self):
        window = SimpleNamespace(
            instrument=SimpleNamespace(RUN_TYPE_MAP={"All": None}),
            run_type_num=None,
            current_run_time="2026-07-20 (Cal)",
            calibration_rb=FakeButton(),
        )

        MainWindow._update_calibration_button_state(window)

        self.assertTrue(window.calibration_rb.enabled)

    def test_non_cal_chunk_remains_disabled_without_run_type_mapping(self):
        window = SimpleNamespace(
            instrument=SimpleNamespace(RUN_TYPE_MAP={"All": None}),
            run_type_num=None,
            current_run_time="2026-07",
            calibration_rb=FakeButton(),
        )

        MainWindow._update_calibration_button_state(window)

        self.assertFalse(window.calibration_rb.enabled)

    def test_explicit_calibration_run_type_still_enables_calibration(self):
        window = SimpleNamespace(
            instrument=SimpleNamespace(RUN_TYPE_MAP={"Calibrations": 2}),
            run_type_num=2,
            current_run_time="2026-07-20 12:00:00",
            calibration_rb=FakeButton(),
        )

        MainWindow._update_calibration_button_state(window)

        self.assertTrue(window.calibration_rb.enabled)


class CATSBatchPortTests(unittest.TestCase):
    def test_update_runs_calculates_air_and_tank_ports_with_week_method(self):
        batch = object.__new__(CATS_batch)
        batch.site = "brw"
        data = pd.DataFrame({
            "analysis_datetime": pd.to_datetime([
                "2026-07-20 01:00Z",
                "2026-07-20 02:00Z",
                "2026-07-20 03:00Z",
                "2026-07-20 04:00Z",
                "2026-07-20 05:00Z",
            ]),
            "port": [2, 4, 6, 8, 99],
            "mf_method_num": [2, 3, 2, 3, 2],
            "height": [1.0, 1.0, 1.0, 1.0, 1.0],
        })

        def calculate(df):
            result = df.copy()
            result["mole_fraction"] = result["mf_method_num"].astype(float)
            return result

        with patch.object(batch, "load_data", return_value=data), \
                patch.object(batch, "calc_mole_fraction", side_effect=calculate):
            result = batch.update_runs(6, channel="q")

        self.assertEqual(result["port"].tolist(), [2, 4, 6, 8])
        self.assertEqual(result["mf_method_num"].tolist(), [3, 3, 3, 3])
        self.assertEqual(result["mole_fraction"].tolist(), [3.0, 3.0, 3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
