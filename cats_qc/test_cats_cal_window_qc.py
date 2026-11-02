import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_cal_window_qc import _windowed_air_outliers


class CatsCalWindowQcTests(unittest.TestCase):
    def setUp(self):
        # 21 days of 2-hourly air readings around 100 with noise of std ~1.
        self.air_times = pd.date_range("2025-01-01", periods=252, freq="2h").to_numpy()
        rng = np.random.default_rng(0)
        self.air_values = 100.0 + rng.normal(scale=1.0, size=252)
        self.air_in_pool = np.ones(252, dtype=bool)
        self.half_window = pd.Timedelta(days=5).to_timedelta64()

    def run_core(self, values, pool=None, sigma=5.0, min_air_points=20):
        return _windowed_air_outliers(
            self.air_times, values,
            self.air_in_pool if pool is None else pool,
            self.half_window, sigma, min_air_points,
        )

    def test_flags_high_and_low_excursions_not_quiet_points(self):
        values = self.air_values.copy()
        values[100] = 110.0
        values[150] = 90.0
        outlier, air_sigma, air_median = self.run_core(values)
        self.assertTrue(outlier[100])
        self.assertTrue(outlier[150])
        self.assertEqual(int(outlier.sum()), 2)
        self.assertTrue(np.isfinite(air_sigma[100]))
        self.assertAlmostEqual(air_median[100], 100.0, delta=0.3)

    def test_bound_is_symmetric(self):
        # The same size of excursion flags on either side.
        high = self.air_values.copy()
        low = self.air_values.copy()
        high[100] = 106.5
        low[100] = 93.5
        self.assertTrue(self.run_core(high)[0][100])
        self.assertTrue(self.run_core(low)[0][100])

    def test_excursion_inside_five_sigma_is_not_flagged(self):
        high = self.air_values.copy()
        low = self.air_values.copy()
        high[100] = 103.0
        low[100] = 97.0
        self.assertFalse(self.run_core(high)[0][100])
        self.assertFalse(self.run_core(low)[0][100])

    def test_too_few_air_points_never_flags(self):
        values = self.air_values.copy()
        values[100] = 200.0
        outlier, air_sigma, air_median = self.run_core(values, min_air_points=1000)
        self.assertFalse(outlier.any())
        self.assertTrue(np.all(np.isnan(air_sigma)))

    def test_no_spread_never_flags(self):
        # Identical neighbours give sigma 0, which is no basis for a verdict.
        values = np.full(252, 100.0)
        values[100] = 110.0
        outlier, air_sigma, air_median = self.run_core(values)
        self.assertFalse(outlier.any())
        self.assertEqual(air_median[100], 100.0)

    def test_leave_one_out_excludes_target_from_its_own_statistics(self):
        # A single extreme point should not be able to pull the median or
        # sigma that judge it far enough to hide its own excursion.
        air_times = pd.date_range("2025-01-01", periods=5, freq="D").to_numpy()
        air_values = np.array([100.0, 101.0, 200.0, 99.0, 100.5])
        outlier, air_sigma, air_median = _windowed_air_outliers(
            air_times, air_values, np.ones(5, dtype=bool),
            pd.Timedelta(days=5).to_timedelta64(), 5.0, 4,
        )
        self.assertEqual(air_median[2], 100.25)
        self.assertTrue(outlier[2])

    def test_already_rejected_neighbors_excluded_from_pool_but_still_scored(self):
        # Index 100 is already rejected for some other reason. It must still
        # be scored as a candidate, but its value can't enter the statistics
        # that judge its neighbours.
        values = self.air_values.copy()
        values[100] = 400.0
        pool = self.air_in_pool.copy()
        pool[100] = False
        pool_all = self.run_core(values)
        pool_excl = self.run_core(values, pool=pool)
        self.assertTrue(pool_excl[0][100])
        clean = self.run_core(self.air_values, pool=pool)
        self.assertEqual(pool_excl[2][99], clean[2][99])
        self.assertNotEqual(pool_all[2][99], pool_excl[2][99])


if __name__ == "__main__":
    unittest.main()
