import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
