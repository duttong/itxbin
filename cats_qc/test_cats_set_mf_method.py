import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_set_mf_method import (
    _OPEN_END,
    _combine_and,
    _coverage_intervals,
    _merge_segments,
    _method_segments,
    _parse_yyyymmdd,
    _port_occupancy_segments,
    _segment_date_filter,
)


def _ts(s):
    return pd.Timestamp(s, tz='UTC')


class FakeCats:
    """Minimal stand-in for CATS_Instrument covering just what the coverage
    helpers touch: port_config_history, scale_assignment_history(), the
    MF_METHOD_* constants, and CAL1_PORT/CAL2_PORT."""

    MF_METHOD_REF = 1
    MF_METHOD_CAL12 = 2
    MF_METHOD_CAL1 = 3
    MF_METHOD_CAL2 = 4
    MF_METHOD_LABELS = {1: 'ref', 2: 'cal12', 3: 'cal1', 4: 'cal2'}
    CAL1_PORT = 2
    CAL2_PORT = 6
    site_num = 15

    def __init__(self, port_rows, assignments=None):
        """port_rows: [(port_num, start_datetime_str, label), ...]
        assignments: {(tank, pnum): [{'start_date':, 'end_date':}, ...]}
        """
        self.port_config_history = pd.DataFrame([
            {'site_num': self.site_num, 'port_num': p,
             'start_datetime': _ts(s), 'label': label}
            for p, s, label in port_rows
        ])
        self._assignments = assignments or {}

    def scale_assignment_history(self, tank, pnum):
        return self._assignments.get((tank, pnum), [])


class ParseDateTests(unittest.TestCase):
    def test_parse_qc_style_date_formats(self):
        self.assertEqual(_parse_yyyymmdd("20260903"), "2026-09-03")
        self.assertEqual(_parse_yyyymmdd("2026-09-03"), "2026-09-03")


class SegmentDateFilterTests(unittest.TestCase):
    def test_open_ended_segment_has_no_upper_bound(self):
        self.assertEqual(
            _segment_date_filter(_ts("2026-01-01"), _OPEN_END),
            "AND a.analysis_time >= '2026-01-01'",
        )

    def test_bounded_segment_has_exclusive_upper_bound(self):
        self.assertEqual(
            _segment_date_filter(_ts("2026-01-01"), _ts("2026-02-01")),
            "AND a.analysis_time >= '2026-01-01'\n"
            "              AND a.analysis_time < '2026-02-01'",
        )


class PortOccupancySegmentsTests(unittest.TestCase):
    def test_two_swaps_tile_the_window_with_no_gaps(self):
        cats = FakeCats(port_rows=[
            (6, "2015-01-01", "TANK_A"),
            (6, "2020-01-01", "TANK_B"),
            (6, "2023-06-01", "TANK_C"),
        ])
        segs = _port_occupancy_segments(cats, 6, _ts("2015-01-01"), _OPEN_END)
        self.assertEqual(
            segs,
            [
                ("TANK_A", _ts("2015-01-01"), _ts("2020-01-01")),
                ("TANK_B", _ts("2020-01-01"), _ts("2023-06-01")),
                ("TANK_C", _ts("2023-06-01"), _OPEN_END),
            ],
        )

    def test_clips_to_requested_window(self):
        cats = FakeCats(port_rows=[
            (6, "2015-01-01", "TANK_A"),
            (6, "2020-01-01", "TANK_B"),
        ])
        segs = _port_occupancy_segments(cats, 6, _ts("2018-01-01"), _ts("2021-01-01"))
        self.assertEqual(
            segs,
            [
                ("TANK_A", _ts("2018-01-01"), _ts("2020-01-01")),
                ("TANK_B", _ts("2020-01-01"), _ts("2021-01-01")),
            ],
        )

    def test_no_history_for_port_returns_empty(self):
        cats = FakeCats(port_rows=[(2, "2015-01-01", "TANK_A")])
        self.assertEqual(_port_occupancy_segments(cats, 6, _ts("2015-01-01"), _OPEN_END), [])


class CoverageIntervalsTests(unittest.TestCase):
    def test_fully_covered_tank(self):
        cats = FakeCats(
            port_rows=[(6, "2015-01-01", "TANK_A")],
            assignments={("TANK_A", 131): [
                {"start_date": "2010-01-01", "end_date": None},
            ]},
        )
        intervals = _coverage_intervals(cats, 6, 131, _ts("2015-01-01"), _OPEN_END)
        self.assertEqual(intervals, [(_ts("2015-01-01"), _OPEN_END, True, "TANK_A")])

    def test_uncovered_tank_with_no_assignments(self):
        cats = FakeCats(port_rows=[(6, "2015-01-01", "TANK_A")])
        intervals = _coverage_intervals(cats, 6, 131, _ts("2015-01-01"), _OPEN_END)
        self.assertEqual(intervals, [(_ts("2015-01-01"), _OPEN_END, False, "TANK_A")])

    def test_mid_occupancy_refill_gap(self):
        # Tank installed 2015-01-01; fill A measured through 2018-01-01, then
        # refilled (fill B) and never measured -- the tail of the occupancy
        # should show up as its own uncovered sub-range.
        cats = FakeCats(
            port_rows=[(6, "2015-01-01", "TANK_A")],
            assignments={("TANK_A", 131): [
                {"start_date": "2015-01-01", "end_date": "2018-01-01"},
            ]},
        )
        intervals = _coverage_intervals(cats, 6, 131, _ts("2015-01-01"), _OPEN_END)
        self.assertEqual(
            intervals,
            [
                (_ts("2015-01-01"), _ts("2018-01-01"), True, "TANK_A"),
                (_ts("2018-01-01"), _OPEN_END, False, "TANK_A"),
            ],
        )


class CombineAndTests(unittest.TestCase):
    def test_both_covered(self):
        cal1 = [(_ts("2015-01-01"), _OPEN_END, True, "T1")]
        cal2 = [(_ts("2015-01-01"), _OPEN_END, True, "T2")]
        out = _combine_and(cal1, cal2)
        self.assertEqual(out, [(_ts("2015-01-01"), _OPEN_END, True, None)])

    def test_one_side_uncovered_names_that_tank(self):
        cal1 = [(_ts("2015-01-01"), _OPEN_END, True, "T1")]
        cal2 = [(_ts("2015-01-01"), _OPEN_END, False, "T2")]
        out = _combine_and(cal1, cal2)
        self.assertEqual(len(out), 1)
        s, e, covered, note = out[0]
        self.assertFalse(covered)
        self.assertIn("cal2 tank T2", note)
        self.assertNotIn("cal1", note)

    def test_misaligned_breakpoints_split_correctly(self):
        cal1 = [
            (_ts("2015-01-01"), _ts("2020-01-01"), True, "T1"),
            (_ts("2020-01-01"), _OPEN_END, False, "T1b"),
        ]
        cal2 = [(_ts("2015-01-01"), _OPEN_END, True, "T2")]
        out = _combine_and(cal1, cal2)
        self.assertEqual(
            [(s, e, c) for s, e, c, n in out],
            [
                (_ts("2015-01-01"), _ts("2020-01-01"), True),
                (_ts("2020-01-01"), _OPEN_END, False),
            ],
        )


class MergeSegmentsTests(unittest.TestCase):
    def test_adjacent_same_method_merges(self):
        segs = [
            (_ts("2015-01-01"), _ts("2018-01-01"), 2, None),
            (_ts("2018-01-01"), _OPEN_END, 2, None),
        ]
        self.assertEqual(
            _merge_segments(segs),
            [(_ts("2015-01-01"), _OPEN_END, 2, [])],
        )

    def test_different_method_does_not_merge(self):
        segs = [
            (_ts("2015-01-01"), _ts("2020-01-01"), 2, None),
            (_ts("2020-01-01"), _OPEN_END, 1, "tank X uncovered"),
        ]
        self.assertEqual(
            _merge_segments(segs),
            [
                (_ts("2015-01-01"), _ts("2020-01-01"), 2, []),
                (_ts("2020-01-01"), _OPEN_END, 1, ["tank X uncovered"]),
            ],
        )

    def test_notes_deduplicated_across_a_merged_run(self):
        segs = [
            (_ts("2015-01-01"), _ts("2016-01-01"), 1, "same note"),
            (_ts("2016-01-01"), _OPEN_END, 1, "same note"),
        ]
        merged = _merge_segments(segs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][3], ["same note"])


class MethodSegmentsRegressionTests(unittest.TestCase):
    """Regression coverage for the real bug: a well-covered historical cal2
    tank record followed by a freshly-swapped, not-yet-measured current
    tank used to force the ENTIRE requested range to ref, discarding the
    historical coverage. It should now only fall back for the uncovered
    tail."""

    def test_current_tank_gap_does_not_poison_historical_coverage(self):
        cats = FakeCats(
            port_rows=[
                (6, "1998-01-01", "OLD_TANK"),
                (6, "2025-11-24", "NEW_TANK"),
            ],
            assignments={("OLD_TANK", 131): [
                {"start_date": "1998-01-01", "end_date": None},
            ]},
            # NEW_TANK has no assignments at all -- unmeasured current fill.
        )
        segments = _method_segments(cats, cats.MF_METHOD_CAL2, 131, _ts("1998-01-01"), _OPEN_END)
        self.assertEqual(
            [(s, e, m) for s, e, m, notes in segments],
            [
                (_ts("1998-01-01"), _ts("2025-11-24"), cats.MF_METHOD_CAL2),
                (_ts("2025-11-24"), _OPEN_END, cats.MF_METHOD_REF),
            ],
        )
        # The fallback segment should explain itself.
        _, _, _, notes = segments[-1]
        self.assertTrue(any("NEW_TANK" in n for n in notes))

    def test_cal12_needs_both_ports_covered(self):
        cats = FakeCats(
            port_rows=[
                (2, "1998-01-01", "CAL1_TANK"),
                (6, "1998-01-01", "CAL2_TANK"),
            ],
            assignments={
                ("CAL1_TANK", 131): [{"start_date": "1998-01-01", "end_date": None}],
                # CAL2_TANK never measured.
            },
        )
        segments = _method_segments(cats, cats.MF_METHOD_CAL12, 131, _ts("1998-01-01"), _OPEN_END)
        self.assertEqual(len(segments), 1)
        s, e, method, notes = segments[0]
        self.assertEqual(method, cats.MF_METHOD_REF)
        self.assertTrue(any("CAL2_TANK" in n for n in notes))

    def test_ref_method_is_never_downgraded(self):
        cats = FakeCats(port_rows=[(6, "1998-01-01", "ANY_TANK")])
        segments = _method_segments(cats, cats.MF_METHOD_REF, 131, _ts("1998-01-01"), _OPEN_END)
        self.assertEqual(segments, [(_ts("1998-01-01"), _OPEN_END, cats.MF_METHOD_REF, [])])


if __name__ == "__main__":
    unittest.main()
