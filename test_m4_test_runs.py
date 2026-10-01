import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from logos_instruments import M4_Instrument
from logosdata.logos_timeseries import TimeseriesWidget
from logosdata.mstar_pairs import MSTAR_PAIR_AVG_SQL


class M4TestPlotting(unittest.TestCase):
    def test_tests_selection_loads_standards_and_separates_samples_on_same_port(self):
        instrument = M4_Instrument.__new__(M4_Instrument)
        instrument.inst_num = 192
        instrument.norm = SimpleNamespace(merge_smoothed_data=lambda df: df)
        records = []
        for i, (label, run_type, port) in enumerate([
            ('t1065_control', 10, 12), ('t1065_201mm', 10, 12),
            ('t1065_control', 10, 12), ('standard', 8, 14),
            ('sx-3574_Arch_A', 7, 15),
        ]):
            records.append(dict(
                analysis_datetime=f'2026-06-18 12:0{i}:00',
                run_time='2026-06-18 09:30:00', sample_datetime=None,
                run_type_num=run_type, detrend_method_num=2, port=port,
                flask_port=0, area=100., net_pressure=10., mole_fraction=1.,
                port_info=label, site=None, pair_id_num=0, sample_id=0,
                rejected=0,
            ))
        instrument.db = Mock()
        instrument.db.doquery.return_value = records
        df = instrument.load_data(22, run_type_num=10,
                                  start_date='2026-06-18', end_date='2026-06-19', verbose=False)
        self.assertEqual(instrument.RUN_TYPE_MAP['Tests'], 10)
        self.assertEqual(df['run_type_num'].tolist(), [10, 10, 10, 8, 7])
        self.assertEqual(df['port_marker'].tolist(), ['X', 'X', 'X', 'D', '^'])
        self.assertNotEqual(df.iloc[0]['port_idx'], df.iloc[1]['port_idx'])
        self.assertEqual(df.iloc[0]['port_idx'], df.iloc[2]['port_idx'])
        self.assertIn('t1065_control', df.iloc[0]['port_label'])

    def test_tests_selection_filters_run_list(self):
        instrument = M4_Instrument.__new__(M4_Instrument)
        instrument.inst_num = 192
        instrument.doquery = Mock(return_value=[{'run_time': '2026-06-18 09:30:00'}])
        self.assertEqual(instrument.query_return_run_list(runtype=10), ['2026-06-18 09:30:00'])
        self.assertIn('AND run_type_num = 10', instrument.doquery.call_args.args[0])

    def test_m4_air_sample_query_excludes_associated_controls_and_type_10(self):
        instrument = SimpleNamespace(inst_num=192, doquery=Mock(return_value=[]))
        harness = SimpleNamespace(
            start_year=SimpleNamespace(value=lambda: 2026),
            end_year=SimpleNamespace(value=lambda: 2026),
            analyte_combo=SimpleNamespace(currentText=lambda: 'CFC12'),
            _resolve_pnum=lambda _: 22, set_current_analyte=lambda _: None,
            _channel_selection=lambda _: (None, False),
            _last_query_params=None, instrument=instrument,
        )
        TimeseriesWidget.query_flask_data(harness)
        sql = instrument.doquery.call_args.args[0]
        self.assertIn('AND test_num = 0 AND run_type_num <> 10', sql)


class MstarAirSampleExport(unittest.TestCase):
    def test_pair_query_excludes_tests_even_with_valid_pair_ids(self):
        conn = sqlite3.connect(':memory:')
        self.addCleanup(conn.close)
        class Std:
            def step(self, value):
                pass
            def finalize(self):
                return 0.
        conn.create_aggregate('STD', 1, Std)
        conn.execute('''CREATE TABLE samples (
            site TEXT, site_num INTEGER, sample_datetime TEXT, inst_num INTEGER,
            inst_id TEXT, pair_id_num INTEGER, parameter_num INTEGER, parameter TEXT,
            wind_speed REAL, wind_direction REAL, channel TEXT, analysis_datetime TEXT,
            analysis_num INTEGER, sample_id INTEGER, value REAL, rejected INTEGER,
            ccgg_event_num INTEGER, test_num INTEGER, run_type_num INTEGER)''')
        for pair, test, run_type in [(1, 0, 1), (2, 1080, 1), (3, 0, 10)]:
            for flask in (1, 2):
                conn.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                             ('BRW', 1, '2026-08-14', 192, 'M4', pair, 22, 'CFC12',
                              0., 0., '', '2026-08-15', pair * 10 + flask, flask,
                              100., 0, 0, test, run_type))
        source = MSTAR_PAIR_AVG_SQL.replace('hats.ng_data_view', 'samples')
        # Adapt only MySQL's display-ID concatenations to SQLite; retain the
        # actual selection, grouping, and two-flask eligibility rules.
        for column in ('analysis_num', 'sample_id'):
            source = source.replace(
                f"GROUP_CONCAT(d.{column} ORDER BY d.{column} SEPARATOR '|')",
                f'GROUP_CONCAT(d.{column})'
            )
        rows = conn.execute(f'SELECT pair_id_num, pair_avg FROM {source} v').fetchall()
        self.assertEqual(rows, [(1, 100.)])


if __name__ == '__main__':
    unittest.main()
