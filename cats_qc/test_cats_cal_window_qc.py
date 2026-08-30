import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_cal_window_qc import _windowed_air_outliers


class CatsCalWindowQcTests(unittest.TestCase):
    def setUp(self):
        # 21 days of daily air readings, quiet around 100 except one high
        # spike (day 10, +10) and one low dip (day 15, -6). Reference tank
        # sampled daily too, steady noise of std ~1.
        self.air_times = pd.date_range("2025-01-01", periods=21, freq="D").to_numpy()
        air_values = np.full(21, 100.0)
        air_values[10] = 110.0   # +10 -> should clear a 3-sigma-high bound (sigma ~1)
        air_values[15] = 94.0    # -6  -> should clear a 2-sigma-low bound
        self.air_values = air_values
        self.air_in_pool = np.ones(21, dtype=bool)

        self.ref_times = pd.date_range("2025-01-01", periods=21, freq="D").to_numpy()
        rng = np.random.default_rng(0)
        self.ref_values = 50.0 + rng.normal(scale=1.0, size=21)
        self.half_window = pd.Timedelta(days=5).to_timedelta64()

    def test_flags_high_and_low_excursions_not_quiet_points(self):
        outlier, ref_std, air_median, n_ref = _windowed_air_outliers(
            self.air_times, self.air_values, self.air_in_pool,
            self.ref_times, self.ref_values,
            self.half_window, sigma_high=3.0, sigma_low=2.0,
            min_ref_points=4, min_air_points=4,
        )
        self.assertTrue(outlier[10])
        self.assertTrue(outlier[15])
        self.assertFalse(outlier[0])
        self.assertFalse(outlier[5])
        self.assertTrue(np.isfinite(ref_std[10]))
        self.assertTrue(np.isfinite(air_median[10]))

    def test_asymmetric_bounds_a_mid_size_dip_is_low_flagged_but_not_high(self):
        # A deviation between 2 and 3 sigma should flag on the low side
        # (sigma_low=2) but not trigger the higher sigma_high=3 bar if it
        # were instead an equal-sized rise.
        values = self.air_values.copy()
        values[10] = 100.0     # remove the high spike from setUp
        values[15] = 97.5      # ~-2.5 sigma given ref noise ~1
        outlier, *_ = _windowed_air_outliers(
            self.air_times, values, self.air_in_pool,
            self.ref_times, self.ref_values,
            self.half_window, sigma_high=3.0, sigma_low=2.0,
            min_ref_points=4, min_air_points=4,
        )
        self.assertTrue(outlier[15])

    def test_too_few_reference_points_never_flags(self):
        outlier, ref_std, air_median, n_ref = _windowed_air_outliers(
            self.air_times, self.air_values, self.air_in_pool,
            self.ref_times, self.ref_values,
            self.half_window, sigma_high=3.0, sigma_low=2.0,
            min_ref_points=1000, min_air_points=4,
        )
        self.assertFalse(outlier.any())
        self.assertTrue(np.all(np.isnan(ref_std)))

    def test_leave_one_out_excludes_target_from_air_median(self):
        # A single extreme point should not be able to pull the median that
        # judges itself far enough to hide its own excursion.
        air_times = pd.date_range("2025-01-01", periods=5, freq="D").to_numpy()
        air_values = np.array([100.0, 100.0, 200.0, 100.0, 100.0])
        air_in_pool = np.ones(5, dtype=bool)
        ref_times = air_times
        ref_values = np.array([50.0, 50.5, 49.8, 50.2, 49.9])
        outlier, ref_std, air_median, n_ref = _windowed_air_outliers(
            air_times, air_values, air_in_pool, ref_times, ref_values,
            pd.Timedelta(days=5).to_timedelta64(),
            sigma_high=3.0, sigma_low=2.0, min_ref_points=4, min_air_points=4,
        )
        self.assertEqual(air_median[2], 100.0)
        self.assertTrue(outlier[2])

    def test_already_rejected_neighbors_excluded_from_pool_but_still_scored(self):
        # Two points (index 1 and 3) are already rejected for some other
        # reason (e.g. cal_step) and sit right next to a candidate (index 2)
        # whose own value is far off. Even though 1 and 3 are excluded from
        # the neighbor pool used to judge OTHERS, they must still themselves
        # be scored as candidates (index 1 clears the high bound on its own
        # merits) -- already_rejected gates pool membership, not candidacy.
        air_times = pd.date_range("2025-01-01", periods=7, freq="D").to_numpy()
        air_values = np.array([100.0, 400.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        air_in_pool = np.array([True, False, True, False, True, True, True])
        ref_times = air_times
        ref_values = np.array([50.0, 50.5, 49.8, 50.2, 49.9, 50.1, 49.95])
        outlier, ref_std, air_median, n_ref = _windowed_air_outliers(
            air_times, air_values, air_in_pool, ref_times, ref_values,
            pd.Timedelta(days=5).to_timedelta64(),
            sigma_high=3.0, sigma_low=2.0, min_ref_points=4, min_air_points=3,
        )
        # index 1 (pool=False) is still evaluated and flagged on its own value.
        self.assertTrue(outlier[1])
        # Its extreme value must not have leaked into index 0's median even
        # though it falls in index 0's window and is chronologically adjacent.
        self.assertEqual(air_median[0], 100.0)


if __name__ == "__main__":
    unittest.main()
