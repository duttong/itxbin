import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_cal_method_qc import _local_level_jump, _period_medians


class PeriodMediansTests(unittest.TestCase):
    def test_median_and_count_per_period(self):
        period_start = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-08"] * 4)
        mole_fraction = pd.Series([10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 100.0])
        rejected = pd.Series([0] * 7)
        out = _period_medians(period_start, mole_fraction, rejected, min_points=3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.loc[0, "median_mf"], 11.0)
        self.assertEqual(out.loc[0, "n"], 3)
        self.assertEqual(out.loc[1, "n"], 4)

    def test_rejected_rows_excluded(self):
        period_start = pd.to_datetime(["2020-01-01"] * 4)
        mole_fraction = pd.Series([10.0, 10.0, 10.0, 999.0])
        rejected = pd.Series([0, 0, 0, 1])
        out = _period_medians(period_start, mole_fraction, rejected, min_points=3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "n"], 3)
        self.assertEqual(out.loc[0, "median_mf"], 10.0)

    def test_below_min_points_dropped(self):
        period_start = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-08"])
        mole_fraction = pd.Series([10.0, 11.0, 20.0])
        rejected = pd.Series([0, 0, 0])
        out = _period_medians(period_start, mole_fraction, rejected, min_points=2)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.loc[0, "median_mf"], 10.5)


def _weekly_index(n, start="2015-01-01"):
    return pd.date_range(start, periods=n, freq="7D")


def _jitter(n, scale=0.05, seed=0):
    # Real mole-fraction data is never perfectly noiseless -- a residual
    # scale of exactly 0.0 (which perfectly linear synthetic data produces)
    # trips the div-by-zero guard in _local_level_jump and forces NaN. Add
    # small jitter, well below any injected step, so tests exercise the
    # normal (nonzero-scale) code path like real data would.
    return np.random.default_rng(seed).normal(scale=scale, size=n)


class LocalLevelJumpTests(unittest.TestCase):
    def test_clean_linear_trend_low_z(self):
        # z < 2.0 for every point in a 40-point range isn't a fair bar under
        # random noise (~5% of normally-distributed points exceed |z|=2 by
        # chance alone) -- check against the production detection threshold
        # instead, which is what actually matters: a clean trend must never
        # cross the bar that would trigger four extra DB round trips.
        t = _weekly_index(80)
        days = np.arange(80) * 7.0
        value = 300.0 + 0.01 * days + _jitter(80)
        jump, z, scale, nb, na = _local_level_jump(
            t.to_numpy(), value, window_days=180, gap_days=14, min_trend_points=6,
        )
        mid = slice(20, 60)
        self.assertTrue(np.all(np.abs(z[mid]) < 4.0), msg=f"max |z| = {np.nanmax(np.abs(z[mid]))}")

    def test_trend_plus_seasonal_cycle_low_z(self):
        # ~3.8 years of weekly data, secular trend + an annual seasonal
        # cycle -- this is the assumption the whole detector depends on: a
        # ~1-year trend window should average the seasonal cycle out of
        # each side's slope, not mistake it for a step. Only score the
        # "safe" middle range where both sides get a FULL (untruncated)
        # window+gap of context -- edge candidates near the array boundary
        # legitimately see an asymmetric partial year and aren't the thing
        # being tested here.
        n = 200
        t = _weekly_index(n)
        days = np.arange(n) * 7.0
        seasonal = 2.0 * np.sin(2 * np.pi * days / 365.25)
        value = 300.0 + 0.01 * days + seasonal + _jitter(n)
        window_days, gap_days = 365.0, 14.0
        jump, z, scale, nb, na = _local_level_jump(
            t.to_numpy(), value, window_days=window_days, gap_days=gap_days, min_trend_points=12,
        )
        margin_weeks = int(np.ceil((window_days + gap_days) / 7.0))
        safe = slice(margin_weeks, n - margin_weeks)
        self.assertGreater(safe.stop - safe.start, 0, "test setup: no safe middle range")
        self.assertTrue(
            np.all(np.abs(z[safe]) < 4.0),
            msg=f"max |z| in safe range = {np.nanmax(np.abs(z[safe]))}",
        )

    def test_injected_step_flagged_with_correct_sign(self):
        # A real step also elevates z for candidates NEAR it, not just at it
        # -- their own before/after window (194 days = window+gap) reaches
        # far enough to straddle the step too, mixing pre-/post-step data
        # into one side's trend fit. That's expected (it's exactly why
        # build_cal_method_qc groups nearby flagged periods into one episode
        # via _group_periods rather than treating each as independent), so
        # only check candidates whose window+gap can't reach the step at all.
        t = _weekly_index(80)
        days = np.arange(80) * 7.0
        value = 300.0 + 0.01 * days + _jitter(80)
        value[40:] += 5.0  # abrupt +5 step at index 40
        jump, z, scale, nb, na = _local_level_jump(
            t.to_numpy(), value, window_days=180, gap_days=14, min_trend_points=6,
        )
        self.assertGreater(z[40], 4.0)
        # NaN (insufficient trailing/leading context near the array edges)
        # is expected and fine here -- only a FINITE score >= 4.0 would be a
        # false positive.
        quiet = np.concatenate([z[0:10], z[70:80]])
        finite_quiet = quiet[np.isfinite(quiet)]
        self.assertGreater(finite_quiet.size, 0, "test setup: no finite quiet-range scores")
        self.assertTrue(np.all(np.abs(finite_quiet) < 4.0), msg=f"{finite_quiet}")

    def test_too_few_points_on_one_side_gives_nan(self):
        # 8 points total: whichever candidate is picked, at most 7 OTHER
        # points exist to split between "before"/"after", so one side
        # always has <= 3 -- below min_trend_points=6 -- regardless of how
        # large window_days is (a big window can't manufacture points that
        # don't exist).
        t = _weekly_index(8)
        value = 300.0 + 0.01 * np.arange(8) * 7.0 + _jitter(8)
        jump, z, scale, nb, na = _local_level_jump(
            t.to_numpy(), value, window_days=180, gap_days=14, min_trend_points=6,
        )
        self.assertTrue(np.all(np.isnan(z)))

    def test_gap_excludes_neighbor_but_window_includes_farther_point(self):
        # Theil-Sen is deliberately robust to outliers, so a single corrupted
        # point among many (e.g. 1 of 25) barely moves the median-of-slopes
        # estimate -- that robustness is a feature, not something to fight in
        # this test. Use small per-side counts (right at min_trend_points) so
        # a single corrupted point actually has visible leverage, making the
        # gap-vs-window contrast measurable.
        t = _weekly_index(30)
        days = np.arange(30) * 7.0
        value = 300.0 + 0.01 * days + _jitter(30)
        window_days, gap_days, min_trend_points = 70.0, 7.0, 6
        candidate = 15  # t_days = 105; 10 points on each side at these settings

        _, z_base, *_ = _local_level_jump(
            t.to_numpy(), value, window_days, gap_days, min_trend_points,
        )

        # index 14 (t_days=98) is within gap_days=7 of the candidate's own
        # time (105) on the "before" side -- excluded from both trend fits,
        # so corrupting it must leave the candidate's score bit-for-bit
        # unchanged.
        corrupted_in_gap = value.copy()
        corrupted_in_gap[14] = 5000.0
        _, z_gap, *_ = _local_level_jump(
            t.to_numpy(), corrupted_in_gap, window_days, gap_days, min_trend_points,
        )
        self.assertEqual(z_gap[candidate], z_base[candidate])

        # index 10 (t_days=70) is well inside the "before" window
        # (>= 105-70-7=28 and < 105-7=98) -- corrupting it must change the
        # candidate's score, even under Theil-Sen's robustness.
        corrupted_in_window = value.copy()
        corrupted_in_window[10] = 5000.0
        _, z_window, *_ = _local_level_jump(
            t.to_numpy(), corrupted_in_window, window_days, gap_days, min_trend_points,
        )
        self.assertGreater(abs(z_window[candidate] - z_base[candidate]), 0.5)


if __name__ == "__main__":
    unittest.main()
