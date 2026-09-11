import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_apply_viterbi import _build_apply_plan


def _period(period_start, chosen):
    return {"period_start": period_start, "chosen": chosen}


class BuildApplyPlanTests(unittest.TestCase):
    def test_single_run_is_open_ended(self):
        df = pd.DataFrame([
            _period("2000-01-01", "cal12"),
            _period("2000-01-08", "cal12"),
            _period("2000-01-15", "cal12"),
        ])
        plan = _build_apply_plan(df)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0], {"method": "cal12", "start_date": "2000-01-01", "end_date": None})

    def test_contiguous_same_method_periods_collapse_into_one_run(self):
        df = pd.DataFrame([
            _period("2000-01-01", "cal12"),
            _period("2000-01-08", "cal12"),
            _period("2000-01-15", "cal2"),
            _period("2000-01-22", "cal2"),
            _period("2000-01-29", "cal12"),
        ])
        plan = _build_apply_plan(df)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0], {"method": "cal12", "start_date": "2000-01-01", "end_date": "2000-01-14"})
        self.assertEqual(plan[1], {"method": "cal2", "start_date": "2000-01-15", "end_date": "2000-01-28"})
        self.assertEqual(plan[2], {"method": "cal12", "start_date": "2000-01-29", "end_date": None})

    def test_a_to_b_to_a_is_three_separate_runs_not_merged(self):
        # Revisiting a method later must NOT merge with its earlier run --
        # each contiguous stretch is its own bounded (or final open-ended)
        # cats_set_mf_method.py call.
        df = pd.DataFrame([
            _period("2000-01-01", "cal12"),
            _period("2000-01-08", "cal2"),
            _period("2000-01-15", "cal12"),
        ])
        plan = _build_apply_plan(df)
        self.assertEqual([s["method"] for s in plan], ["cal12", "cal2", "cal12"])
        self.assertEqual(plan[0]["end_date"], "2000-01-07")
        self.assertEqual(plan[1]["end_date"], "2000-01-14")
        self.assertEqual(plan[2]["end_date"], None)

    def test_out_of_order_input_is_sorted_by_period_start(self):
        df = pd.DataFrame([
            _period("2004-03-01", "cal12"),
            _period("2001-01-01", "cal2"),
        ])
        plan = _build_apply_plan(df)
        self.assertEqual([s["start_date"] for s in plan], ["2001-01-01", "2004-03-01"])

    def test_mid_week_transition_timestamp_end_date_uses_calendar_day(self):
        # A period_start with a time-of-day component (a mid-week tank
        # swap split) must not leave a gap: the previous run's end_date is
        # the calendar day before, regardless of the time-of-day the next
        # run actually starts at.
        df = pd.DataFrame([
            _period("2000-01-01", "cal12"),
            _period("2000-01-08 16:10:00", "cal2"),
        ])
        plan = _build_apply_plan(df)
        self.assertEqual(plan[0]["end_date"], "2000-01-07")
        self.assertEqual(plan[1]["start_date"], "2000-01-08")

    def test_single_row(self):
        df = pd.DataFrame([_period("2000-01-01", "cal12")])
        plan = _build_apply_plan(df)
        self.assertEqual(plan, [{"method": "cal12", "start_date": "2000-01-01", "end_date": None}])


if __name__ == "__main__":
    unittest.main()
