import unittest

import pandas as pd

from cats_gcwerks2db import CATS_GCwerks2DB


class FakeDB:
    def __init__(self):
        self.inserts = []
        self.queries = []

    def doMultiInsert(self, sql, params, all=False):
        self.inserts.append((sql, list(params), all))

    def doquery(self, sql, params=None):
        self.queries.append((sql, list(params or [])))
        return []


class CATSInvalidMeasurementTests(unittest.TestCase):
    def make_importer(self):
        importer = object.__new__(CATS_GCwerks2DB)
        importer.analytes = {"CFC11": 114}
        importer.flagged = False
        importer.db = FakeDB()
        return importer

    def test_empty_measurement_requires_every_metric_to_be_missing_like(self):
        self.assertTrue(CATS_GCwerks2DB._empty_measurement(0, 0.0, None))
        self.assertTrue(CATS_GCwerks2DB._empty_measurement(pd.NA, None, 0))
        self.assertFalse(CATS_GCwerks2DB._empty_measurement(0, 1.0, 0))

    def test_invalid_rows_delete_tags_and_measurement_but_valid_row_is_upserted(self):
        importer = self.make_importer()
        df = pd.DataFrame({
            "analysis_time_str": ["valid", "zero", "null"],
            "CFC11_f_ht": [100.0, 0.0, pd.NA],
            "CFC11_f_area": [200.0, 0.0, pd.NA],
            "CFC11_f_rt": [300.0, 0.0, pd.NA],
        })
        analysis_map = {"valid": 1, "zero": 2, "null": 3}

        importer.upsert_mole_fractions(df, analysis_map)

        self.assertEqual(len(importer.db.inserts), 1)
        _sql, params, use_all = importer.db.inserts[0]
        self.assertTrue(use_all)
        self.assertEqual(params, [(1, 114, "f", 100.0, 200.0, 300.0)])

        self.assertEqual(len(importer.db.queries), 2)
        tag_sql, tag_params = importer.db.queries[0]
        measurement_sql, measurement_params = importer.db.queries[1]
        self.assertIn("DELETE t FROM hats.ng_insitu_mole_fraction_tags", tag_sql)
        self.assertIn("DELETE m FROM hats.ng_insitu_mole_fractions", measurement_sql)
        self.assertEqual(tag_params, [2, 114, "f", 3, 114, "f"])
        self.assertEqual(measurement_params, tag_params)
        self.assertNotIn("ng_insitu_analysis", tag_sql + measurement_sql)


if __name__ == "__main__":
    unittest.main()
