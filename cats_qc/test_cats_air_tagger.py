import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_air_tagger import _rolling_median, detect_air_excursions


def _series(n, start="2000-01-01", freq="8h", seed=0, scale=0.3):
    t = pd.date_range(start, periods=n, freq=freq).to_numpy()
    days = np.arange(n) * (pd.Timedelta(freq) / pd.Timedelta(days=1))
    seasonal = 2.0 * np.sin(2 * np.pi * days / 365.25)
    trend = 300.0 + 0.01 * days
    value = trend + seasonal + np.random.default_rng(seed).normal(scale=scale, size=n)
    return t, value


class RollingMedianTests(unittest.TestCase):
    def test_flat_series_reproduces_constant(self):
        t = pd.date_range("2020-01-01", periods=50, freq="1D").to_numpy()
        v = np.full(50, 10.0)
        out = _rolling_median(t, v, window_days=20, min_points=4)
        mid = slice(15, 35)
        self.assertTrue(np.allclose(out[mid], 10.0))

    def test_below_min_points_is_nan(self):
        t = pd.date_range("2020-01-01", periods=5, freq="1D").to_numpy()
        v = np.arange(5, dtype=float)
        out = _rolling_median(t, v, window_days=100, min_points=10)
        self.assertTrue(np.all(np.isnan(out)))

    def test_nan_neighbors_excluded(self):
        t = pd.date_range("2020-01-01", periods=6, freq="1D").to_numpy()
        v = np.array([1.0, np.nan, 1.0, 1.0, np.nan, 1.0])
        out = _rolling_median(t, v, window_days=10, min_points=3)
        # candidate index 3 (value 1.0): neighbours are indices 0,1,2,4,5 with
        # values [1.0, nan, 1.0, nan, 1.0] -> 3 finite neighbours, median 1.0
        self.assertAlmostEqual(out[3], 1.0)


class DetectAirExcursionsTests(unittest.TestCase):
    def test_clean_series_rarely_false_positive(self):
        # A hard zero-false-positives bar isn't fair on ~1100 tested points
        # at sigma=4 -- occasional single points near the threshold by
        # chance alone are expected (same lesson as
        # test_cats_cal_method_qc.py's LocalLevelJumpTests), so check the
        # RATE stays low rather than demanding none at all.
        t, v = _series(1500, seed=1)
        result = detect_air_excursions(
            t, v, median_window_days=45, mad_window_days=45, min_points=8, sigma=4.0,
        )
        mid = slice(200, 1300)  # away from edge effects
        rate = result["outlier"][mid].sum() / (mid.stop - mid.start)
        self.assertLess(rate, 0.01, msg=f"false-positive rate = {rate}")

    def test_injected_excursion_flagged_no_false_positives(self):
        t, v = _series(2000, seed=0)
        v = v.copy()
        v[500:530] += 6.0  # 10-day excursion at 8h cadence, well above noise
        result = detect_air_excursions(
            t, v, median_window_days=45, mad_window_days=45, min_points=8, sigma=4.0,
        )
        outlier = result["outlier"]
        self.assertEqual(outlier[500:530].sum(), 30)
        self.assertEqual(outlier.sum(), 30)  # nothing flagged outside the injected block

    def test_excursion_much_longer_than_window_defeats_detection(self):
        # Documents the algorithm's own stated limitation: an excursion
        # spanning most/all of the median window is no longer a minority of
        # the window's points, so the median gets pulled toward it instead
        # of staying at the true background level -- same breakdown-point
        # caveat as cats_baseline_qc.py's/cats_cal_window_qc.py's local
        # references.
        t, v = _series(3000, seed=2)
        v = v.copy()
        v[500:2500] += 6.0  # excursion far longer than a 45-day window
        result = detect_air_excursions(
            t, v, median_window_days=45, mad_window_days=45, min_points=8, sigma=4.0,
        )
        # The interior of the long excursion is NOT flagged (median tracks
        # it as if it were the new normal) -- only its edges show up as
        # transitions, if anything given noise. This asserts the known
        # limitation rather than a desired outcome.
        interior = result["outlier"][1000:2000]
        self.assertEqual(interior.sum(), 0)


if __name__ == "__main__":
    unittest.main()
