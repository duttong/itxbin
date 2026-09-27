import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from logosdata.logos_compare.logos_compare import (
    LogosCompareWindow,
    ProgramSelection,
    _average_instrument_monthly_means,
    _combine_monthly_mean_frames,
)


class _Value:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def doquery(self, sql, params):
        self.calls.append((sql, params))
        return self.rows


class LogosCompareQueryTests(unittest.TestCase):
    def test_combined_monthly_means_are_pair_weighted(self):
        fe3 = pd.DataFrame(
            [{"site": "BRW", "month_start": "2000-01-01", "monthly_avg": 100.0,
              "monthly_std": 2.0, "n": 2}]
        )
        otto = pd.DataFrame(
            [{"site": "BRW", "month_start": "2000-01-01", "monthly_avg": 110.0,
              "monthly_std": 4.0, "n": 3}]
        )

        result = _combine_monthly_mean_frames([fe3, otto])

        self.assertEqual(result.iloc[0]["n"], 5)
        self.assertEqual(result.iloc[0]["monthly_avg"], 106.0)
        self.assertAlmostEqual(result.iloc[0]["monthly_std"], np.sqrt(39.0))

    def test_otto_query_uses_pair_info_without_flask_id(self):
        db = _FakeDB(
            [{"site": "BRW", "month_start": "1998-01-01", "monthly_avg": 315.2,
              "monthly_std": 0.4, "n": 2}]
        )
        harness = SimpleNamespace(
            start_year=_Value(1998),
            end_year=_Value(1998),
            loaders={"fe3": SimpleNamespace(instrument=db)},
        )
        selection = ProgramSelection("fecd", "N2O", 5)

        result = LogosCompareWindow._query_otto_monthly_mean_data(
            harness, selection, ["BRW"]
        )

        sql, params = db.calls[0]
        self.assertIn("JOIN hats.hatsflask_pair_info pi ON pi.pair_id = v.pair_id_num", sql)
        self.assertNotIn("flask_id", sql)
        self.assertEqual(params, [5, "BRW", 1998, 1998])
        self.assertEqual(result.iloc[0]["site"], "BRW")
        self.assertEqual(result.iloc[0]["month_start"], pd.Timestamp("1998-01-01"))

    def test_fecd_combines_fe3_and_otto_months(self):
        fe3 = pd.DataFrame(
            [{"site": "BRW", "month_start": "2000-01-01", "monthly_avg": 100.0,
              "monthly_std": 2.0, "n": 2}]
        )
        otto = pd.DataFrame(
            [{"site": "BRW", "month_start": "2000-01-01", "monthly_avg": 110.0,
              "monthly_std": 4.0, "n": 3}]
        )
        loader = SimpleNamespace(_preferred_channel_filter_sql=lambda *_args: "")
        harness = SimpleNamespace(
            loaders={"fe3": loader},
            _sql_condition_from_and_filter=lambda sql: sql,
            _query_combined_pair_monthly_mean_data=lambda **_kwargs: fe3,
            _query_otto_monthly_mean_data=lambda *_args: otto,
        )

        result = LogosCompareWindow._query_fecd_monthly_mean_data(
            harness, ProgramSelection("fecd", "N2O", 5), ["BRW"]
        )

        self.assertEqual(result.iloc[0]["n"], 5)
        self.assertEqual(result.iloc[0]["monthly_avg"], 106.0)


class InsituCombineTests(unittest.TestCase):
    def test_overlapping_instruments_get_equal_weight(self):
        rits = pd.DataFrame(
            [{"site": "BRW", "month_start": pd.Timestamp("1998-10-01"), "monthly_avg": 100.0,
              "monthly_std": 3.0, "n": 700}]
        )
        cats = pd.DataFrame(
            [{"site": "BRW", "month_start": pd.Timestamp("1998-10-01"), "monthly_avg": 110.0,
              "monthly_std": 4.0, "n": 50},
             {"site": "BRW", "month_start": pd.Timestamp("1998-11-01"), "monthly_avg": 111.0,
              "monthly_std": 2.0, "n": 60}]
        )

        result = _average_instrument_monthly_means([rits, cats, pd.DataFrame()])

        overlap = result.iloc[0]
        self.assertEqual(overlap["monthly_avg"], 105.0)
        self.assertAlmostEqual(overlap["monthly_std"], np.sqrt(12.5))
        self.assertEqual(overlap["n"], 750)
        self.assertEqual(result.iloc[1]["monthly_avg"], 111.0)
        self.assertEqual(len(result), 2)

    def test_all_empty_returns_empty_frame(self):
        result = _average_instrument_monthly_means([pd.DataFrame(), pd.DataFrame()])
        self.assertTrue(result.empty)
        self.assertIn("monthly_avg", result.columns)

    def test_rits_query_maps_sites_to_inst_nums(self):
        db = _FakeDB(
            [{"site": "BRW", "month_start": "1990-01-01", "monthly_avg": 308.0,
              "monthly_std": 0.5, "n": 600}]
        )
        harness = SimpleNamespace(
            start_year=_Value(1987),
            end_year=_Value(2001),
            loaders={"ie3": SimpleNamespace(instrument=db)},
        )

        result = LogosCompareWindow._query_rits_monthly_mean_data(
            harness, ProgramSelection("insitu", "N2O", 5), ["BRW", "SUM", "SPO"]
        )

        sql, params = db.calls[0]
        self.assertIn("hats.ng_insitu_monthly_means", sql)
        self.assertEqual(params, [246, 250, 5, 1987, 2001])
        self.assertEqual(result.iloc[0]["month_start"], pd.Timestamp("1990-01-01"))

    def test_rits_query_skips_sites_without_rits(self):
        db = _FakeDB([])
        harness = SimpleNamespace(
            start_year=_Value(1987), end_year=_Value(2001),
            loaders={"ie3": SimpleNamespace(instrument=db)},
        )

        result = LogosCompareWindow._query_rits_monthly_mean_data(
            harness, ProgramSelection("insitu", "N2O", 5), ["SUM"]
        )

        self.assertTrue(result.empty)
        self.assertEqual(db.calls, [])
