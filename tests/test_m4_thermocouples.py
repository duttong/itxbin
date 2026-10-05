import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import pandas as pd

from logosdata import m4_thermocouples as thermocouples


class BdFileSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_uses_filename_date_across_years_and_ignores_mtime(self):
        older = self.directory / "bd123125.txt"
        newer = self.directory / "bd010126.txt"
        older.touch()
        newer.touch()
        os.utime(newer, (0, 0))
        self.assertEqual(thermocouples.latest_bd_file(self.directory), newer)

    def test_ignores_invalid_dates_other_names_and_directories(self):
        valid = self.directory / "bd100226.txt"
        for name in (valid.name, "bd023126.txt", "bd999999.txt", "bd100326.xl", "bd100426extra.txt"):
            (self.directory / name).touch()
        (self.directory / "bd100526.txt").mkdir()
        self.assertEqual(thermocouples.latest_bd_file(self.directory), valid)

    def test_no_valid_logs_has_clear_error(self):
        with self.assertRaisesRegex(SystemExit, "No valid bdMMDDYY.txt files"):
            thermocouples.latest_bd_file(self.directory)

    def test_main_defaults_to_newest_log_and_output_name(self):
        latest = self.directory / "bd100226.txt"
        latest.touch()
        with patch.object(thermocouples, "GSPC_DIR", self.directory), patch.object(thermocouples, "make_figure") as plot, patch("sys.argv", ["m4_thermocouples.py"]):
            thermocouples.main()
        plot.assert_called_once_with(latest, "bd100226_thermocouples.png")

    def test_explicit_bare_name_and_output_override_are_preserved(self):
        explicit = self.directory / "bd100126.txt"
        explicit.touch()
        (self.directory / "bd100226.txt").touch()
        with patch.object(thermocouples, "GSPC_DIR", self.directory), patch.object(thermocouples, "make_figure") as plot, patch("sys.argv", ["m4_thermocouples.py", "bd100126", "-o", "custom.png"]):
            thermocouples.main()
        plot.assert_called_once_with(explicit, "custom.png")

    def test_selected_run_uses_month_day_year_and_archive(self):
        archive = self.directory / "2026"
        archive.mkdir()
        selected = archive / "bd100226.txt"
        selected.touch()
        self.assertEqual(
            thermocouples.bd_file_for_run(self.directory, "2026-10-02 11:30:00 (Cal)"),
            selected,
        )
        active = self.directory / selected.name
        active.touch()
        self.assertEqual(
            thermocouples.bd_file_for_run(self.directory, "2026-10-02 11:30:00"),
            active,
        )

    def test_missing_selected_run_does_not_use_latest_file(self):
        (self.directory / "bd100226.txt").touch()
        with self.assertRaisesRegex(FileNotFoundError, "bd100126.txt"):
            thermocouples.bd_file_for_run(self.directory, "2026-10-01 09:00:00")

    def test_missing_temperature_files_has_clear_error(self):
        with self.assertRaisesRegex(SystemExit, "No temps_.*csv files found"):
            thermocouples.read_temps(self.directory, None, None)

    def test_temperature_parsing_excludes_time_only_copies_without_warnings(self):
        (self.directory / "temps_original.csv").write_text(
            "datetime,therm0,therm1\n"
            "2026-01-21 23:59:59,20,-170\n"
            "2026-01-22 00:00:00,21,-171\n"
            "2026-01-22 00:00:01.500,22,-172\n"
            "invalid,23,-173\n"
        )
        (self.directory / "temps_copy.csv").write_text(
            "datetime,therm0,therm1\n15:21:04,99,99\n7:23:42,99,99\n"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            temps = thermocouples.read_temps(
                self.directory, pd.Timestamp("2026-01-21 23:59:59"),
                pd.Timestamp("2026-01-22 00:00:02"),
            )
        self.assertEqual(temps.datetime.tolist(), [
            pd.Timestamp("2026-01-21 23:59:59"),
            pd.Timestamp("2026-01-22 00:00:00"),
            pd.Timestamp("2026-01-22 00:00:01.500"),
        ])
        self.assertEqual(temps.therm0.tolist(), [20, 21, 22])

    def test_figure_panel_heights_are_one_to_two(self):
        bd = self.directory / "bd100226.txt"
        bd.write_text(
            "2026-10-02 00:00:00,000: Start\n"
            "2026-10-02 00:04:00,000: Sample valve open\n"
            "2026-10-02 00:08:00,000: Sample valve closed\n"
            "2026-10-02 00:24:00,000: End\n"
        )
        pd.DataFrame({
            "datetime": pd.date_range("2026-10-02", periods=145, freq="10s"),
            "therm0": 20.0, "therm1": -170.0,
        }).to_csv(self.directory / "temps_2026-10-02.csv", index=False)
        figure, injections = thermocouples.build_figure(bd)
        self.assertEqual(injections, 1)
        upper, lower = figure.axes
        self.assertAlmostEqual(lower.get_position().height / upper.get_position().height, 2)


if __name__ == "__main__":
    unittest.main()
