"""Tests for flask-pair queries used by the LOGOS data agent."""

import unittest

from logosdata.logos_agent_tools import LOGOSDataAgentTools


class _FakeDb:
    def __init__(self):
        self.queries = []

    def doquery(self, sql, params):
        self.queries.append((sql, params))
        if "FROM hats.hatsflask_event_view v" in sql and "hatsflask_inv" in sql:
            return [{
                "pair_id_num": 42,
                "site_code": "BRW",
                "Flask_1": "1001",
                "Flask_2": "1002",
                "Flask_Type": "S",
                "Sample_Date": "2026-08-03",
                "sample_datetime_utc": "2026-08-03 18:49:00",
                "Wind_Speed": 9.4,
                "Wind_Direction": 85.0,
                "Air_Temp": -2.0,
                "Dew_Point": -4.0,
                "Precipitation": "1",
                "Sky": "2",
                "Comments": "clear",
                "Operator": "RB",
            }]
        if "FROM hats.hatsflask_event_view v" in sql:
            return [{
                "pair_id_num": 42,
                "site": "BRW",
                "sample_datetime": "2026-08-03 18:49:00",
                "Flask_1": "1001",
                "Flask_2": "1002",
            }]
        if "FROM hats.ng_pair_avg_view" in sql:
            return []
        raise AssertionError(f"Unexpected SQL:\n{sql}")


class FlaskPairQueryTests(unittest.TestCase):
    def setUp(self):
        self.db = _FakeDb()
        self.tools = object.__new__(LOGOSDataAgentTools)
        self.tools.db = self.db
        self.tools.inst_id = "fe3"
        self.tools.inst_num = 193

    def test_recent_pairs_use_hatsflask_event_view(self):
        pairs = self.tools.get_recent_flask_pairs("brw", limit=1)

        self.assertEqual(pairs["rows"][0]["sample_ids"], [1001, 1002])
        query = self.db.queries[0][0]
        self.assertIn("FROM hats.hatsflask_event_view v", query)
        self.assertNotIn("Status_MetData", query)

    def test_pair_metadata_uses_hatsflask_event_view_and_inventory(self):
        metadata = self.tools.get_pair_metadata(42)["pair_metadata"]

        self.assertEqual(metadata["sample_ids"], [1001, 1002])
        self.assertEqual(metadata["flask_type"], "S")
        self.assertEqual(metadata["air_temp"], -2.0)
        query = self.db.queries[0][0]
        self.assertIn("FROM hats.hatsflask_event_view v", query)
        self.assertIn("LEFT JOIN hats.hatsflask_inv", query)
        self.assertNotIn("Status_MetData", query)


if __name__ == "__main__":
    unittest.main()
