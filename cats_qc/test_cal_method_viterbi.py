import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cal_method_viterbi import (
    build_candidate_table,
    robust_step_scale,
    solve_viterbi,
    tank_transition_mask,
    total_variation,
)


def _ts(s):
    return pd.Timestamp(s, tz='UTC')


class RobustStepScaleTests(unittest.TestCase):
    def test_constant_series_has_zero_scale(self):
        self.assertEqual(robust_step_scale(np.array([1.0, 1.0, 1.0, 1.0])), 0.0)

    def test_scales_with_typical_step_size(self):
        # Steady +1.0 steps -> MAD of (diffs - median(diffs)) is 0, so scale
        # is 0 even though the series itself is far from flat -- the metric
        # measures step IRREGULARITY, not step size.
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(robust_step_scale(values), 0.0)

    def test_one_irregular_step_gives_nonzero_scale(self):
        values = np.array([0.0, 1.0, 1.0, 5.0, 5.0])
        self.assertGreater(robust_step_scale(values), 0.0)

    def test_ignores_nan(self):
        values = np.array([0.0, np.nan, 1.0, np.nan, 2.0])
        # Same as dense [0,1,2] once NaN is dropped -> constant diffs -> 0
        self.assertEqual(robust_step_scale(values), 0.0)

    def test_too_few_points_returns_zero(self):
        self.assertEqual(robust_step_scale(np.array([1.0])), 0.0)
        self.assertEqual(robust_step_scale(np.array([])), 0.0)


class TotalVariationTests(unittest.TestCase):
    def test_sums_absolute_steps(self):
        self.assertAlmostEqual(total_variation(np.array([1.0, 3.0, 2.0, 5.0])), 2 + 1 + 3)

    def test_ignores_nan(self):
        self.assertAlmostEqual(
            total_variation(np.array([1.0, np.nan, 3.0, np.nan, 2.0])), 2 + 1,
        )

    def test_fewer_than_two_points_is_nan(self):
        self.assertTrue(np.isnan(total_variation(np.array([1.0]))))
        self.assertTrue(np.isnan(total_variation(np.array([]))))


class TankTransitionMaskTests(unittest.TestCase):
    class FakeBatch:
        CAL1_PORT = 2
        CAL2_PORT = 6

        def __init__(self, cal1_series, cal2_series):
            self._cal1 = cal1_series
            self._cal2 = cal2_series

        def tank_serials_for_dates(self, port_num, dates):
            return self._cal1 if port_num == self.CAL1_PORT else self._cal2

    def test_flags_only_the_period_after_a_swap(self):
        periods = pd.Series([_ts("2020-01-01"), _ts("2020-01-08"), _ts("2020-01-15")])
        cal1 = pd.Series(["T1", "T1", "T2"])  # swaps between period 1 and 2
        cal2 = pd.Series(["S1", "S1", "S1"])  # never swaps
        mask = tank_transition_mask(self.FakeBatch(cal1, cal2), periods)
        np.testing.assert_array_equal(mask, [False, False, True])

    def test_no_swaps_all_false(self):
        periods = pd.Series([_ts("2020-01-01"), _ts("2020-01-08"), _ts("2020-01-15")])
        cal1 = pd.Series(["T1", "T1", "T1"])
        cal2 = pd.Series(["S1", "S1", "S1"])
        mask = tank_transition_mask(self.FakeBatch(cal1, cal2), periods)
        np.testing.assert_array_equal(mask, [False, False, False])

    def test_either_port_swapping_flags_the_period(self):
        periods = pd.Series([_ts("2020-01-01"), _ts("2020-01-08")])
        cal1 = pd.Series(["T1", "T1"])
        cal2 = pd.Series(["S1", "S2"])  # cal2 swaps, cal1 doesn't
        mask = tank_transition_mask(self.FakeBatch(cal1, cal2), periods)
        np.testing.assert_array_equal(mask, [False, True])


class SolveViterbiTests(unittest.TestCase):
    """No DB, no CATS_batch -- pure function over a hand-built table."""

    def _table(self, rows):
        df = pd.DataFrame(rows, columns=["period_start", "cal12", "cal2", "cal1"])
        df["period_mid"] = df["period_start"]
        return df

    def test_prefers_cal12_when_all_candidates_agree(self):
        rows = [(_ts(f"2020-01-{i+1:02d}"), 100.0 + i, 100.0 + i, 100.0 + i) for i in range(5)]
        table = self._table(rows)
        pref = {"cal12": 0.0, "cal2": 0.5, "cal1": 1.0}
        mask = np.zeros(len(table), dtype=bool)
        solved = solve_viterbi(table, pref, mask, low_switch_cost=1.0, high_switch_cost=5.0)
        self.assertTrue((solved["chosen"] == "cal12").all())

    def test_switches_when_cal12_has_a_real_persistent_jump(self):
        # cal12 steps up by 20 and never returns; cal2 stays flat throughout.
        rows = [
            (_ts("2020-01-01"), 100.0, 100.0, np.nan),
            (_ts("2020-01-08"), 100.5, 100.4, np.nan),
            (_ts("2020-01-15"), 120.0, 100.6, np.nan),  # cal12 jumps here
            (_ts("2020-01-22"), 120.3, 100.5, np.nan),
            (_ts("2020-01-29"), 120.1, 100.7, np.nan),
        ]
        table = self._table(rows)
        pref = {"cal12": 0.0, "cal2": 0.1, "cal1": 0.2}
        mask = np.zeros(len(table), dtype=bool)
        solved = solve_viterbi(table, pref, mask, low_switch_cost=1.0, high_switch_cost=2.0)
        # A sustained ~20-unit gap should be worth switching for, even at
        # the "not a tank transition" (high) switch cost.
        self.assertEqual(solved["chosen"].iloc[-1], "cal2")

    def test_cheap_switch_cost_at_a_tank_transition(self):
        rows = [
            (_ts("2020-01-01"), 100.0, 99.0, np.nan),
            (_ts("2020-01-08"), 100.2, 99.1, np.nan),
            (_ts("2020-01-15"), 200.0, 99.3, np.nan),  # transition: cal12 breaks
        ]
        table = self._table(rows)
        pref = {"cal12": 0.0, "cal2": 0.1, "cal1": 0.2}
        mask = np.array([False, False, True])
        # High switch cost would make this not worth it; low switch cost
        # (paid because this period is a tank transition) makes it cheap.
        solved = solve_viterbi(table, pref, mask, low_switch_cost=0.5, high_switch_cost=1000.0)
        self.assertEqual(solved["chosen"].iloc[-1], "cal2")

    def test_missing_candidate_periods_are_bridged(self):
        # cal1 is the only candidate available at period 2 (e.g. cal12/cal2
        # both lacked coverage that week) -- the DP must still produce a
        # full path rather than raising.
        rows = [
            (_ts("2020-01-01"), 100.0, 100.0, np.nan),
            (_ts("2020-01-08"), np.nan, np.nan, 100.5),
            (_ts("2020-01-15"), 100.4, 100.3, np.nan),
        ]
        table = self._table(rows)
        pref = {"cal12": 0.0, "cal2": 0.1, "cal1": 0.2}
        mask = np.zeros(len(table), dtype=bool)
        solved = solve_viterbi(table, pref, mask, low_switch_cost=1.0, high_switch_cost=5.0)
        self.assertEqual(len(solved), 3)
        self.assertEqual(solved["chosen"].iloc[1], "cal1")
        self.assertTrue(np.isfinite(solved["chosen_mf"]).all())

    def test_single_row_table(self):
        table = self._table([(_ts("2020-01-01"), 100.0, 99.0, np.nan)])
        pref = {"cal12": 0.0, "cal2": 0.1, "cal1": 0.2}
        mask = np.zeros(1, dtype=bool)
        solved = solve_viterbi(table, pref, mask, low_switch_cost=1.0, high_switch_cost=5.0)
        self.assertEqual(solved["chosen"].iloc[0], "cal12")


class BuildCandidateTableSmoothingTests(unittest.TestCase):
    """Exercise just the rolling-median smoothing step build_candidate_table
    applies, without a DB -- construct the pre-smoothing table directly and
    reuse the same rolling().median() call the function makes."""

    def test_lone_outlier_is_smoothed_away(self):
        s = pd.Series([10.0, 10.1, 50.0, 10.2, 10.0])
        smoothed = s.rolling(3, center=True, min_periods=1).median()
        self.assertLess(smoothed.iloc[2], 20.0)

    def test_persistent_shift_survives_smoothing(self):
        s = pd.Series([10.0, 10.1, 30.0, 30.1, 30.2, 30.0])
        smoothed = s.rolling(3, center=True, min_periods=1).median()
        self.assertGreater(smoothed.iloc[-1], 25.0)


if __name__ == "__main__":
    unittest.main()
