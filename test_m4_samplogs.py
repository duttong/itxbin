import contextlib
import io
import datetime
import re
import sqlite3
import unittest
from unittest.mock import Mock

import pandas as pd

from m4_samplogs import M4_SampleLogs, M4_Serial_Numbers, parse_run_index_sample_info, parse_test_num


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
        self.assertEqual([(row[5], row[3]) for row in rows],
                         [(label, 10 if i < 3 else run_type)
                          for i, (label, run_type) in enumerate(labels)])
        pd.testing.assert_frame_equal(frame, original)
        sql = self.ingest.db.doMultiInsert.call_args.args[0]
        self.assertIn('test_num     = IF(VALUES(test_num) = 0', sql)
        self.assertIn('AND port_info = VALUES(port_info)', sql)
        # Compare the original label before the upsert replaces port_info.
        self.assertLess(sql.index('test_num     = IF'), sql.index('port_info    = VALUES'))
        self.assertEqual(sql.count('%s'), len(rows[0]))

    def test_unknown_numbers_warn_and_can_be_resolved_on_reimport(self):
        frame = self.frame([('T1099_control', 1)])
        self.ingest.db.doquery.return_value = None
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.ingest.insert_ng_analysis(frame)
        self.assertIn('M4 test 1099', output.getvalue())
        self.assertEqual(self.inserted_rows()[0][-1], 0)
        self.assertEqual(self.inserted_rows()[0][3], 1)
        self.ingest.db.doquery.return_value = [{'test_num': 1099}]
        self.ingest.insert_ng_analysis(frame)
        self.assertEqual(self.inserted_rows()[0][-1], 1099)
        self.assertEqual(self.inserted_rows()[0][3], 10)

    def test_ordinary_samples_do_not_query_equipment_tests(self):
        self.ingest.insert_ng_analysis(self.frame([('SX-3531', 8), ('12-1234', 5), ('sx-3574_Arch_A', 7)]))
        self.ingest.db.doquery.assert_not_called()
        self.assertEqual([row[-1] for row in self.inserted_rows()], [0, 0, 0])
        self.assertEqual([row[3] for row in self.inserted_rows()], [8, 5, 7])

    def test_serial_invalidated_only_when_sample_identity_changes(self):
        self.ingest.insert_ng_analysis(self.frame([('sx-3534_Grav_D', 7)]))
        sql = self.ingest.db.doMultiInsert.call_args.args[0]
        self.assertLess(sql.index('tank_serial_num = IF'), sql.index('run_type_num = VALUES'))
        self.assertLess(sql.index('tank_serial_num = IF'), sql.index('port_info    = VALUES'))
        # Evaluate the actual upsert expression against old/new identities.
        # SQLite's IS implements MySQL's NULL-safe <=> comparison.
        expression = sql.split('tank_serial_num = IF(', 1)[1].split('NULL)', 1)[0] + 'NULL'
        expression = re.sub(r'VALUES[(]([a-z_]+)[)]',
                            lambda match: ':new_' + match[1], expression).replace('<=>', 'IS')
        conn = sqlite3.connect(':memory:')
        self.addCleanup(conn.close)
        conn.create_function('IF', 3, lambda condition, yes, no: yes if condition else no)
        for old_label, new_label, old_type, new_type, expected in [
            ('sx-3593_Arch_B', 'sx-3534_Grav_D', 7, 7, None),
            ('sx-3593_Arch_B', 'sx-3593_Arch_B', 7, 7, 'SX-3593'),
            ('sx-3593_Arch_B', 'sx-3593_Arch_B', 7, 10, None),
            (None, 'sx-3534_Grav_D', 7, 7, None),
        ]:
            with self.subTest(old_label=old_label, new_label=new_label, new_type=new_type):
                result = conn.execute(
                    f'SELECT IF({expression}) FROM '
                    '(SELECT ? AS port_info, ? AS run_type_num, ? AS tank_serial_num)',
                    # Named values also bind positionally, in order of appearance.
                    (new_label, new_type, old_label, old_type, 'SX-3593'),
                ).fetchone()[0]
                self.assertEqual(result, expected)

    def test_validation_failure_prevents_any_writes(self):
        self.ingest.db.doquery.side_effect = RuntimeError('database unavailable')
        with self.assertRaisesRegex(RuntimeError, 'database unavailable'):
            self.ingest.insert_ng_analysis(self.frame([('T1073_7099', 1)]))
        self.ingest.db.doMultiInsert.assert_not_called()


class TestSerialRefresh(unittest.TestCase):
    def setUp(self):
        self.sn = M4_Serial_Numbers.__new__(M4_Serial_Numbers)
        self.sn.inst_num = 192
        self.sn.start_date = '19940718'
        self.sn._fill_serial_cache = {}
        self.sn.db = Mock()

    def test_default_query_keeps_missing_only_scope(self):
        self.sn.db.doquery.return_value = []
        self.sn.find_missing_serials()
        sql = self.sn.db.doquery.call_args.args[0]
        self.assertIn('and tank_serial_num is NULL', sql)
        self.assertNotIn('OR (run_time', sql)
        self.assertEqual(len(self.sn.db.doquery.call_args.args), 1)

    def test_refresh_repairs_swapped_serials_including_overnight_injections(self):
        rows = [
            dict(num=3230807, analysis_time='2026-11-20 01:34:00',
                 run_time='2026-11-19 15:11:00', run_type_num=7,
                 port_info='sx-3534_Grav_D', tank_serial_num='SX-3593'),
            dict(num=3230815, analysis_time='2026-11-20 04:32:00',
                 run_time='2026-11-19 15:11:00', run_type_num=7,
                 port_info='sx-3593_Arch_B', tank_serial_num='SX-3534'),
            dict(num=3230791, run_type_num=7,
                 port_info='sx-3593_Arch_A', tank_serial_num='SX-3593'),
            dict(num=999, run_type_num=7,
                 port_info='unresolved', tank_serial_num='MANUAL'),
        ]
        def query(sql, params=None):
            if 'SELECT * FROM hats.ng_analysis' in sql:
                self.assertIn('OR (run_time >= %s AND run_time < %s)', sql)
                self.assertEqual(params, (datetime.date(2026, 11, 19), datetime.date(2026, 11, 20)))
                # Exercise MySQLdb's parameter formatting, including LIKE wildcards.
                formatted = sql % tuple(str(value) for value in params)
                self.assertIn('not like "%zero%"', formatted)
                self.assertIn('not like "%junk%"', formatted)
                return rows
            if 'FROM reftank.fill' in sql:
                return [{'serial_number': 'SX-3534' if 'sx-3534' in sql else 'SX-3593'}]
            return []
        self.sn.db.doquery.side_effect = query
        self.sn.update_serials(datetime.date(2026, 11, 19))
        updates = [call.args[0] for call in self.sn.db.doquery.call_args_list
                   if call.args[0].startswith('UPDATE')]
        self.assertEqual(len(updates), 2)
        self.assertIn("SET tank_serial_num = 'SX-3534'", updates[0])
        self.assertIn('num = 3230807 AND inst_num = 192', updates[0])
        self.assertIn("SET tank_serial_num = 'SX-3593'", updates[1])
        self.assertIn('num = 3230815 AND inst_num = 192', updates[1])
        lookups = [call for call in self.sn.db.doquery.call_args_list
                   if 'FROM reftank.fill' in call.args[0]]
        self.assertEqual(len(lookups), 2)  # One lookup per distinct tank key.


class TestCcgFlaskEvents(unittest.TestCase):
    """CCGG flasks run on M4 as flasks under their own id (GRF firn air)."""

    def setUp(self):
        self.ingest = M4_SampleLogs.__new__(M4_SampleLogs)
        self.ingest.db = Mock()

    @staticmethod
    def row(tank, site='GRF', sample_time='260512', samptype='flask', pair_id=0):
        return {'samptype': samptype, 'tank': tank, 'site': site,
                'sample_time': sample_time, 'pair_id': pair_id}

    @staticmethod
    def event(num, flask_id='C14#11', date='2026-05-12'):
        return {'num': num, 'id': flask_id, 'date': pd.Timestamp(date).date(), 'site': 'GRF'}

    def test_label_parses_hash_ids(self):
        for label, tank in [('grf_12_may_26_#C14#11', 'C14#11'),
                            ('grf_23_may_26_#SIO-22', 'SIO-22'),
                            ('brw_16_oct_26_#5612-31199', '5612-31199')]:
            with self.subTest(label=label):
                self.assertEqual(parse_run_index_sample_info(label)['tank'], tank)

    def test_site_date_match(self):
        self.ingest.db.doquery.return_value = [self.event(567872)]
        self.assertEqual(self.ingest.fetch_ccgg_event_num(self.row('C14#11')), 567872)
        sql, params = self.ingest.db.doquery.call_args.args
        self.assertIn('e.id = %s', sql)
        self.assertEqual(params, ('C14#11', 'GRF', '2026-05-12', '2026-05-12'))

    def test_one_day_off_match(self):
        self.ingest.db.doquery.return_value = [self.event(567872, date='2026-05-13')]
        self.assertEqual(self.ingest.fetch_ccgg_event_num(self.row('C14#11')), 567872)

    def test_ambiguous_or_missing_is_empty(self):
        for rows in ([], [self.event(1, date='2026-05-11'), self.event(2, date='2026-05-13')]):
            with self.subTest(rows=rows):
                self.ingest.db.doquery.return_value = rows
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertIsNone(self.ingest.fetch_ccgg_event_num(self.row('C14#11')))

    def test_hats_pairs_and_tanks_are_not_looked_up(self):
        for row in (self.row('5612-31199', site='BRW'),          # HATS flask-pair label
                    self.row('C14#11', pair_id=31199),           # has a HATS pair
                    self.row('C14#11', samptype='std'),
                    self.row('1234'),                            # bare number
                    self.row('C14#11', site=None),               # no site/date in label
                    self.row('C14#11', sample_time=None)):
            with self.subTest(row=row):
                self.assertIsNone(self.ingest.fetch_ccgg_event_num(row))
        self.ingest.db.doquery.assert_not_called()


if __name__ == '__main__':
    unittest.main()
