import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from logos_instruments_core import HATS_DB_Functions
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


class LegacyScaleAssignmentTests(unittest.TestCase):
    def test_n2o_falls_back_to_gas_specific_reference_table(self):
        instrument = object.__new__(HATS_DB_Functions)
        instrument.db = SimpleNamespace(doquery=Mock(side_effect=[[], [], [
            {'serial_number': 'ALM-033782', 'start_date': '1995-01-01',
             'coef0': 311.42, 'coef1': 0, 'coef2': 0,
             'standard_unc': 0.2, 'level': 'Primary'},
        ]]))

        result = instrument.scale_assignments('ALM-033782', 5, run_date='1998-01-01')

        self.assertEqual(result['coef0'], 311.42)
        self.assertEqual(instrument.db.doquery.call_args_list[2].args[0].split('FROM ')[1].split()[0],
                         'reftank.N2O_X2006A')

    def test_legacy_history_has_same_shape_as_view_history(self):
        instrument = object.__new__(HATS_DB_Functions)
        instrument.db = SimpleNamespace(doquery=Mock(return_value=[
            {'serial_number': 'ALM-024307', 'start_date': '1999-05-11',
             'coef0': 4.403, 'standard_unc': 0.01, 'level': 'Primary'},
        ]))

        result = instrument.legacy_scale_assignment_history('ALM-024307', 6)

        self.assertEqual(result[0]['coef0'], 4.403)
        self.assertIsNone(result[0]['fill_code'])

class EmptyPlotTests(unittest.TestCase):
    def test_gc_plot_clears_previous_figure_when_analyte_has_no_data(self):
        window = SimpleNamespace(
            run=pd.DataFrame(),
            clear_plot=Mock(),
        )

        MainWindow.gc_plot(window, "mole_fraction")

        window.clear_plot.assert_called_once_with()


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
        for inst_id in ("cats", "ie3", "fe3"):
            with self.subTest(inst_id=inst_id):
                instrument = SimpleNamespace(
                    inst_id=inst_id,
                    preferred_channels_for_range=lambda _pnum, _start, _end: {"q"},
                )
                window = SimpleNamespace(
                    instrument=instrument,
                    analytes={"N2O (q)": 5, "N2O (a)": 5},
                )

                preferred, tooltip = MainWindow._preferred_analyte_label(
                    window, "N2O (q)",
                    pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31")
                )
                other, _ = MainWindow._preferred_analyte_label(
                    window, "N2O (a)",
                    pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31")
                )

                self.assertEqual(preferred, "N2O (q) ★")
                self.assertEqual(other, "N2O (a)")
                self.assertIn("final preferred-channel product", tooltip)

    def test_preferred_marker_range_matches_instrument_run_selection(self):
        ie3_window = SimpleNamespace(
            instrument=SimpleNamespace(inst_id="ie3"),
            current_run_time="2026-07",
        )
        fe3_window = SimpleNamespace(
            instrument=SimpleNamespace(inst_id="fe3"),
            current_run_time="2026-07-20 12:34:56 (Cal)",
        )

        ie3_start, ie3_end = MainWindow._preferred_marker_range(ie3_window)
        fe3_start, fe3_end = MainWindow._preferred_marker_range(fe3_window)

        self.assertEqual(ie3_start, pd.Timestamp("2026-07-01"))
        self.assertEqual(ie3_end, pd.Timestamp("2026-07-31 23:59:59"))
        self.assertEqual(fe3_start, pd.Timestamp("2026-07-20 12:34:56"))
        self.assertEqual(fe3_end, fe3_start)

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
    @staticmethod
    def _port_history():
        return pd.DataFrame({
            "start_datetime": pd.to_datetime([
                "2023-01-01 00:00Z", "2023-01-01 00:00Z", "2026-03-24 01:15Z",
            ]),
            "site_num": [1, 1, 1],
            "port_num": [2, 6, 6],
            "label": ["CAL1", "OLD_REF", "NEW_REF"],
        })

    def test_tank_serial_resolution_uses_configuration_date(self):
        instrument = object.__new__(CATS_Instrument)
        instrument.site_num = 1
        instrument.port_config_history = self._port_history()
        instrument.port_config = pd.DataFrame({
            "site_num": [1, 1], "port_num": [2, 6],
            "label": ["CAL1", "NEW_REF"],
        })

        serials = instrument.tank_serials_for_dates(6, pd.Series(pd.to_datetime([
            "2025-07-01 00:00Z", "2026-03-25 00:00Z",
        ])))

        self.assertEqual(serials.tolist(), ["OLD_REF", "NEW_REF"])
        self.assertIsNone(instrument.tank_serial_for_port(6, "2022-01-01"))

    def test_scale_simple_uses_reference_tank_installed_on_each_row(self):
        instrument = object.__new__(CATS_Instrument)
        instrument.site_num = 1
        instrument.port_config_history = self._port_history()
        instrument.port_config = pd.DataFrame({
            "site_num": [1], "port_num": [6], "label": ["NEW_REF"],
        })
        histories = {
            "OLD_REF": [{"start_date": pd.Timestamp("2023-01-01").date(),
                         "end_date": None, "coef0": 336.0, "unc_c0": 0.1}],
            "NEW_REF": [{"start_date": pd.Timestamp("2025-01-01").date(),
                         "end_date": None, "coef0": 339.0, "unc_c0": 0.1}],
        }
        instrument.scale_assignment_history = Mock(
            side_effect=lambda serial, _pnum: histories[serial]
        )
        data = pd.DataFrame({
            "analysis_datetime": pd.to_datetime([
                "2026-03-23 00:00Z", "2026-03-25 00:00Z",
            ]),
            "parameter_num": [5, 5],
            "normalized_resp": [1.0, 1.0],
        })

        result = instrument.calc_mole_fraction_scale_simple(data)

        self.assertEqual(result["mole_fraction"].tolist(), [336.0, 339.0])

    def test_update_fits_uses_each_weeks_installed_reference_tank(self):
        batch = object.__new__(CATS_batch)
        batch.site = "spo"
        batch.site_num = 1
        batch.inst_num = 244
        batch.port_config_history = self._port_history()
        batch.port_config = pd.DataFrame({
            "site_num": [1, 1], "port_num": [2, 6],
            "label": ["CAL1", "NEW_REF"],
        })
        data = pd.DataFrame({
            "analysis_datetime": pd.to_datetime([
                "2026-03-16 01:00Z", "2026-03-16 02:00Z",
                "2026-03-30 01:00Z", "2026-03-30 02:00Z",
            ]),
            "port": [2, 6, 2, 6],
            "normalized_resp": [0.9, 1.0, 0.9, 1.0],
            "rejected": [0, 0, 0, 0],
        })
        histories = {
            "CAL1": [{"start_date": pd.Timestamp("2023-01-01").date(),
                      "end_date": None, "coef0": 300.0, "unc_c0": 0.1}],
            "OLD_REF": [{"start_date": pd.Timestamp("2023-01-01").date(),
                         "end_date": None, "coef0": 336.0, "unc_c0": 0.1}],
            "NEW_REF": [{"start_date": pd.Timestamp("2025-01-01").date(),
                         "end_date": None, "coef0": 339.0, "unc_c0": 0.1}],
        }
        batch.load_data = Mock(return_value=data)
        batch.get_week_mf_method = Mock(return_value=batch.MF_METHOD_CAL12)
        batch.scale_assignment_history = Mock(
            side_effect=lambda serial, _pnum: histories[serial]
        )
        batch._resolve_scale_num = Mock(return_value=7)

        fits, scale_num, _ref_serial, channel = batch.update_fits(
            5, channel="q", start_date="2026-03-01", end_date="2026-04-01"
        )

        self.assertEqual(scale_num, 7)
        self.assertEqual(channel, "q")
        self.assertEqual(fits["ref_serial"].tolist(), ["OLD_REF", "NEW_REF"])

    def test_update_fits_skips_week_that_contains_a_tank_change(self):
        batch = object.__new__(CATS_batch)
        batch.site = "spo"
        batch.site_num = 1
        batch.inst_num = 244
        batch.port_config_history = self._port_history()
        batch.port_config = pd.DataFrame({
            "site_num": [1, 1], "port_num": [2, 6],
            "label": ["CAL1", "NEW_REF"],
        })
        batch.load_data = Mock(return_value=pd.DataFrame({
            "analysis_datetime": pd.to_datetime([
                "2026-03-23 00:30Z", "2026-03-23 01:00Z", "2026-03-25 01:00Z",
            ]),
            "port": [2, 6, 6],
            "normalized_resp": [0.9, 1.0, 1.0],
            "rejected": [0, 0, 0],
        }))
        batch.get_week_mf_method = Mock(return_value=batch.MF_METHOD_CAL12)
        batch.scale_assignment_history = Mock(return_value=[{
            "start_date": pd.Timestamp("2023-01-01").date(),
            "end_date": None, "coef0": 300.0, "unc_c0": 0.1,
        }])

        fits, *_ = batch.update_fits(
            5, channel="q", start_date="2026-03-23", end_date="2026-03-29"
        )

        self.assertTrue(fits.empty)

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
