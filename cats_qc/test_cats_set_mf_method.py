import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_set_mf_method import _date_filter, _parse_yyyymmdd


class CatsSetMfMethodDateTests(unittest.TestCase):
    def test_parse_qc_style_date_formats(self):
        self.assertEqual(_parse_yyyymmdd("20260903"), "2026-09-03")
        self.assertEqual(_parse_yyyymmdd("2026-09-03"), "2026-09-03")

    def test_date_filter_is_open_ended_without_end(self):
        self.assertEqual(
            _date_filter("2026-01-01"),
            "AND a.analysis_time >= '2026-01-01'",
        )

    def test_date_filter_includes_requested_end_date(self):
        self.assertEqual(
            _date_filter("2026-01-01", "2026-01-31"),
            "AND a.analysis_time >= '2026-01-01'\n"
            "              AND a.analysis_time < DATE_ADD('2026-01-31', INTERVAL 1 DAY)",
        )


if __name__ == "__main__":
    unittest.main()
