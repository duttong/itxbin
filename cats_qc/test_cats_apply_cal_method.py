import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_apply_cal_method import _build_apply_plan, _retag_groups


def _episode(episode_start, recommendation, gas="SF6", channel="q", pnum=6, **extra):
    row = {
        "episode_start": episode_start, "gas": gas, "channel": channel, "pnum": pnum,
        "recommendation": recommendation,
    }
    row.update(extra)
    return row


class BuildApplyPlanTests(unittest.TestCase):
    def test_apply_steps_carry_start_date_and_method(self):
        df = pd.DataFrame([
            _episode("2001-01-01", "cal2"),
            _episode("2002-06-15", "cal1"),
        ])
        plan = _build_apply_plan(df)
        self.assertEqual([s["action"] for s in plan], ["apply", "apply"])
        self.assertEqual(plan[0]["start_date"], "2001-01-01")
        self.assertEqual(plan[0]["method"], "cal2")
        self.assertEqual(plan[1]["start_date"], "2002-06-15")
        self.assertEqual(plan[1]["method"], "cal1")

    def test_out_of_order_input_is_sorted_by_episode_start(self):
        df = pd.DataFrame([
            _episode("2004-03-01", "cal12"),
            _episode("2001-01-01", "cal2"),
        ])
        plan = _build_apply_plan(df)
        self.assertEqual([s["start_date"] for s in plan], ["2001-01-01", "2004-03-01"])

    def test_unresolved_episode_is_skipped_not_applied(self):
        df = pd.DataFrame([
            _episode("2001-01-01", "cal2"),
            _episode("2002-06-15", "UNRESOLVED", cal1_tank="T1", cal2_tank="T2",
                     anchor_period_mid="2002-09-01"),
            _episode("2004-03-01", "cal12"),
        ])
        plan = _build_apply_plan(df)
        self.assertEqual([s["action"] for s in plan], ["apply", "skip", "apply"])
        skip_step = plan[1]
        self.assertEqual(skip_step["cal1_tank"], "T1")
        self.assertEqual(skip_step["cal2_tank"], "T2")

    def test_pnum_is_cast_to_int_for_apply_steps(self):
        df = pd.DataFrame([_episode("2001-01-01", "cal2", pnum="6")])
        plan = _build_apply_plan(df)
        self.assertEqual(plan[0]["pnum"], 6)
        self.assertIsInstance(plan[0]["pnum"], int)


class RetagGroupsTests(unittest.TestCase):
    def test_one_group_per_gas_channel_at_earliest_applied_start(self):
        df = pd.DataFrame([
            _episode("2004-03-01", "cal12"),
            _episode("2001-01-01", "cal2"),
            _episode("2002-06-15", "UNRESOLVED"),
        ])
        plan = _build_apply_plan(df)
        groups = _retag_groups(plan)
        self.assertEqual(groups, [{"gas": "SF6", "channel": "q", "start_date": "2001-01-01"}])

    def test_unresolved_only_gas_produces_no_retag_group(self):
        df = pd.DataFrame([_episode("2001-01-01", "UNRESOLVED")])
        plan = _build_apply_plan(df)
        self.assertEqual(_retag_groups(plan), [])

    def test_separate_groups_per_gas_and_per_channel(self):
        df = pd.DataFrame([
            _episode("2001-01-01", "cal2", gas="SF6", channel="q", pnum=6),
            _episode("2003-01-01", "cal1", gas="N2O", channel="q", pnum=5),
            _episode("2005-01-01", "cal12", gas="N2O", channel="a", pnum=5),
        ])
        plan = _build_apply_plan(df)
        groups = _retag_groups(plan)
        self.assertEqual(groups, [
            {"gas": "N2O", "channel": "a", "start_date": "2005-01-01"},
            {"gas": "N2O", "channel": "q", "start_date": "2003-01-01"},
            {"gas": "SF6", "channel": "q", "start_date": "2001-01-01"},
        ])


if __name__ == "__main__":
    unittest.main()
