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
from logos_timeseries import TimeseriesWidget


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


class PreferredChannelDisplayTests(unittest.TestCase):
    def test_preferred_channels_for_range_handles_transition(self):
        instrument = object.__new__(CATS_Instrument)
        instrument.preferred_channel_history = pd.DataFrame([
            {"parameter_num": 22, "start_date": "2017-03-01", "channel": "a"},
            {"parameter_num": 22, "start_date": "2021-05-20", "channel": "f"},
        ])

        before = instrument.preferred_channels_for_range(
            22, "2021-05-01", "2021-05-19 23:59:59"
        )
        crossing = instrument.preferred_channels_for_range(
            22, "2021-05-01", "2021-05-31 23:59:59"
        )

        self.assertEqual(before, {"a"})
        self.assertEqual(crossing, {"a", "f"})

    def test_processing_star_does_not_change_analyte_identity(self):
        instrument = SimpleNamespace(
            inst_id="cats",
            preferred_channels_for_range=lambda _pnum, _start, _end: {"q"},
        )
        window = SimpleNamespace(
            instrument=instrument,
            analytes={"N2O (q)": 5, "N2O (a)": 5},
        )

        preferred, tooltip = MainWindow._preferred_analyte_label(
            window, "N2O (q)", pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31")
        )
        other, _ = MainWindow._preferred_analyte_label(
            window, "N2O (a)", pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31")
        )

        self.assertEqual(preferred, "N2O (q) ★")
        self.assertEqual(other, "N2O (a)")
        self.assertIn("final preferred-channel product", tooltip)

    def test_plot_name_uses_the_selected_real_channel(self):
        window = SimpleNamespace(
            analytes={"N2O (q)": 5, "N2O (a)": 5},
            current_pnum=5,
            current_channel="q",
            instrument=SimpleNamespace(analytes_inv={5: "N2O (a)"}),
        )

        name = MainWindow._current_analyte_name(window)

        self.assertEqual(name, "N2O (q)")

    def test_cats_timeseries_preferred_filter_is_date_aware(self):
        class FakeTimeseries:
            _uses_forced_preferred_channel = TimeseriesWidget._uses_forced_preferred_channel

        widget = FakeTimeseries()
        widget.force_preferred_channel = True
        widget.instrument = SimpleNamespace(inst_num=239, return_preferred_channel=lambda: None)

        sql = TimeseriesWidget._preferred_channel_filter_sql(
            widget,
            "mf.channel", "mf.parameter_num", "a.analysis_time"
        )

        self.assertIn("pc.start_date <= a.analysis_time", sql)
        self.assertIn("pc.inst_num = 239", sql)

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
