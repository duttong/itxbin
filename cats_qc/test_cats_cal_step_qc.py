import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_cal_step_qc import _group_periods, _port_rate_outliers


class CatsCalStepQcTests(unittest.TestCase):
    def test_port_rate_outliers_flags_only_the_step_point(self):
        times = pd.date_range("1999-04-01", periods=20, freq="2h", tz="UTC")
        # Small jitter around 100, a sudden step to ~150, then jitter around 150.
        heights = [100.0, 100.5, 99.8, 100.3, 99.9, 100.2, 100.1, 99.7, 100.4, 100.0,
                   150.2, 149.8, 150.5, 149.9, 150.3, 150.0, 149.7, 150.4, 150.1, 149.8]
        sub = pd.DataFrame({"analysis_datetime": times, "height": heights})
        # scale_window_days/min_periods scaled down to fit this short synthetic
        # span -- production defaults (30 days, 20 obs) need a longer series.
        result = _port_rate_outliers(
            sub, mad_multiplier=3.5, scale_window_days=2.0, scale_min_periods=5
        )
        self.assertFalse(result.loc[5, "rate_outlier"])
        self.assertTrue(result.loc[10, "rate_outlier"])
        self.assertFalse(result.loc[15, "rate_outlier"])

    def test_group_periods_bridges_small_gaps_and_splits_large_ones(self):
        times = pd.to_datetime([
            "1999-04-09 21:43", "1999-04-09 23:44", "1999-04-10 01:44",
            "1999-04-15 00:00",
        ], utc=True)
        periods = _group_periods(pd.Series(times), max_gap_hours=24.0)
        self.assertEqual(len(periods), 2)
        self.assertEqual(periods[0], (times[0], times[2]))
        self.assertEqual(periods[1], (times[3], times[3]))

    def test_group_periods_empty_input(self):
        self.assertEqual(_group_periods(pd.Series([], dtype="datetime64[ns, UTC]"), 24.0), [])


if __name__ == "__main__":
    unittest.main()
