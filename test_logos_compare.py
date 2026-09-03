import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from logosdata.logos_compare.logos_compare import (
    LogosCompareWindow,
    ProgramSelection,
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
