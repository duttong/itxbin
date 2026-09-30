import contextlib
import io
import unittest
from unittest.mock import Mock

import pandas as pd

from m4_samplogs import M4_SampleLogs, M4_Serial_Numbers, parse_test_num


class TestLabels(unittest.TestCase):
    def test_four_digit_prefixes(self):
        for label, expected in [
            ('T1073_DT0050853', 1073),
            ('t1057_bld_21_apr_26_#7036-1798', 1057),
            (' T1065-control ', 1065), ('T1080', 1080),
            ('T1073 sample', 1073),
            ('T107_7099', 0), ('T10730_7099', 0),
            ('T1073abc', 0), ('DT0050853', 0), ('12-1234', 0),
            ('prefix_T1073_7099', 0), ('', 0),
            (None, 0), (float('nan'), 0), (pd.NA, 0),
        ]:
            with self.subTest(label=label):
                self.assertEqual(parse_test_num(label), expected)

    def test_serial_lookup_ignores_test_number(self):
        serials = M4_Serial_Numbers.__new__(M4_Serial_Numbers)
        for label, expected in [
            ('T1073_DT0050853', '0050853'),
            ('t1062_DT0049855_w_reg', '0049855'),
            ('t1080_sx-3574', 'sx-3574'),
            ('T1080_ESX_3574', 'sx-3574'),
            ('T1065_control', None), ('T1080', None),
            ('SX-3531', 'sx-3531'), ('ALM-064967', '064967'),
            ('12-1234', '1234'), ('', None), (None, None),
        ]:
            with self.subTest(label=label):
                self.assertEqual(serials.port_key(label), expected)


class TestAnalysisIngest(unittest.TestCase):
    def setUp(self):
        self.ingest = M4_SampleLogs.__new__(M4_SampleLogs)
        self.ingest.inst_num = 192
        self.ingest.db = Mock()
        self.ingest.db.doMultiInsert.return_value = False

    @staticmethod
    def frame(labels):
        return pd.DataFrame([
            dict(dt_run=pd.Timestamp('2026-09-30 12:00') + pd.Timedelta(minutes=i),
                 run_time=pd.Timestamp('2026-09-30 11:00'),
                 run_type_num=run_type, ssvpos='12', tank=label,
                 pair_id=0, flask_id=0, ccgg_event_num=None)
            for i, (label, run_type) in enumerate(labels)
        ])

    def inserted_rows(self):
        return self.ingest.db.doMultiInsert.call_args.args[1]

    def test_validates_distinct_numbers_and_preserves_labels_and_types(self):
        self.ingest.db.doquery.return_value = [{'test_num': 1073}, {'test_num': 1065}]
        labels = [('T1073_7099', 1), ('t1073_DT0050853', 1),
                  ('t1065_control', 1), ('SX-3531', 8), ('12-1234', 5)]
        frame = self.frame(labels)
        original = frame.copy(deep=True)
        self.ingest.insert_ng_analysis(frame)
        query, params = self.ingest.db.doquery.call_args.args
        self.assertIn('SELECT DISTINCT test_num FROM ccgg_equip.equip_tests_view', query)
        self.assertEqual(params, (1065, 1073))
        self.ingest.db.doquery.assert_called_once()
        rows = self.inserted_rows()
        self.assertEqual([row[-1] for row in rows], [1073, 1073, 1065, 0, 0])
        self.assertEqual([(row[5], row[3]) for row in rows], labels)
        pd.testing.assert_frame_equal(frame, original)
        sql = self.ingest.db.doMultiInsert.call_args.args[0]
        self.assertIn('test_num     = VALUES(test_num)', sql)
        self.assertEqual(sql.count('%s'), len(rows[0]))

    def test_unknown_numbers_warn_and_can_be_resolved_on_reimport(self):
        frame = self.frame([('T1099_control', 1)])
        self.ingest.db.doquery.return_value = None
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.ingest.insert_ng_analysis(frame)
        self.assertIn('M4 test 1099', output.getvalue())
        self.assertEqual(self.inserted_rows()[0][-1], 0)
        self.ingest.db.doquery.return_value = [{'test_num': 1099}]
        self.ingest.insert_ng_analysis(frame)
        self.assertEqual(self.inserted_rows()[0][-1], 1099)

    def test_ordinary_samples_do_not_query_equipment_tests(self):
        self.ingest.insert_ng_analysis(self.frame([('SX-3531', 8), ('12-1234', 5)]))
        self.ingest.db.doquery.assert_not_called()
        self.assertEqual([row[-1] for row in self.inserted_rows()], [0, 0])

    def test_validation_failure_prevents_any_writes(self):
        self.ingest.db.doquery.side_effect = RuntimeError('database unavailable')
        with self.assertRaisesRegex(RuntimeError, 'database unavailable'):
            self.ingest.insert_ng_analysis(self.frame([('T1073_7099', 1)]))
        self.ingest.db.doMultiInsert.assert_not_called()


if __name__ == '__main__':
    unittest.main()
