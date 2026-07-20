import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gcwerks_import import GCwerks_Import
from ie3_import import IE3_import
from itx_import import ITX


class GCwerksImportTests(unittest.TestCase):
    def make_options(self):
        return argparse.Namespace(
            year=2026,
            smoothfile=None,
            s=False,
            ws_start=-1,
            boxwidth=None,
            g=False,
            SGwin=61,
            SGorder=4,
        )

    def test_import_itx_skips_too_small_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itx_file = Path(tmpdir) / 'bad.itx'
            itx_file.write_text('short\n')

            importer = GCwerks_Import('smo', self.make_options())

            with patch('gcwerks_import.run') as run_mock:
                result = importer.import_itx(itx_file)

            self.assertFalse(result)
            run_mock.assert_not_called()

    def test_import_itx_compresses_and_skips_when_hook_requests_skip(self):
        class SkipImporter(GCwerks_Import):
            def should_skip_itx(self, itx):
                return True

        class FakeITX:
            def __init__(self):
                self.data = ['ok']
                self.chroms = [[1, 2]]
                self.compress_to_Z_calls = []

            def compress_to_Z(self, path):
                self.compress_to_Z_calls.append(path)

        fake_itx = FakeITX()
        importer = SkipImporter('smo', self.make_options())
        itx_file = Path('/tmp/skipme.itx.gz')

        with patch('gcwerks_import.itx_import.ITX', return_value=fake_itx), \
                patch('gcwerks_import.run') as run_mock:
            result = importer.import_itx(itx_file)

        self.assertFalse(result)
        self.assertEqual(fake_itx.compress_to_Z_calls, [itx_file])
        run_mock.assert_not_called()


class IE3ImportTests(unittest.TestCase):
    def make_args(self):
        return argparse.Namespace(
            year=2026,
            smoothfile=None,
            s=False,
            ws_start=-1,
            boxwidth=11,
            g=False,
            SGwin=81,
            SGorder=4,
            source='/tmp/source',
            dest_root='/tmp/dest',
            past_days=2,
            target_date=None,
        )

    def test_remove_last_itx_only_touches_requested_day_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            day1 = base / '20260325'
            day2 = base / '20260326'
            day1.mkdir()
            day2.mkdir()

            old_file = day1 / 'older.itx.gz'
            kept_file = day1 / 'keep.itx.gz'
            other_day_file = day2 / 'other.itx.gz'
            old_file.write_text('a' * 200)
            kept_file.write_text('b' * 200)
            other_day_file.write_text('c' * 200)
            os.utime(old_file, (1, 1))
            os.utime(kept_file, (2, 2))
            os.utime(other_day_file, (3, 3))

            importer = IE3_import('smo', self.make_args())
            importer._remove_last_itx(day1)

            self.assertFalse(kept_file.exists())
            self.assertTrue(old_file.exists())
            self.assertTrue(other_day_file.exists())

    def test_should_skip_itx_matches_skip_flag_in_note(self):
        importer = IE3_import('smo', self.make_args())

        itx = object.__new__(ITX)
        itx.data = [
            'IGOR',
            'X note chr1_00001, " 1; 0; 06:56:07; 03-25-2026; 5; smo; SKIP; 1013.06;"',
            'X SetScale /P x, 0, 0.20, chr1_00001',
        ]

        self.assertTrue(importer.should_skip_itx(itx))

    def test_should_not_skip_itx_without_skip_flag(self):
        importer = IE3_import('smo', self.make_args())

        itx = object.__new__(ITX)
        itx.data = [
            'IGOR',
            'X note chr1_00001, " 1; 0; 06:56:07; 03-25-2026; 5; smo; KEEP; 1013.06;"',
            'X SetScale /P x, 0, 0.20, chr1_00001',
        ]

        self.assertFalse(importer.should_skip_itx(itx))


class CalibrationFakeDB:
    def __init__(self):
        self.query_calls = []
        self.multi_insert_calls = []

    def doquery(self, sql, params=None):
        self.query_calls.append((sql, params))
        if "SELECT formula FROM gmd.parameter" in sql:
            return [{"formula": "CH2I2"}]
        return []

    def doMultiInsert(self, sql, params, all=False):
        self.multi_insert_calls.append((sql, list(params), all))
        return False


class CalibrationCleanupTests(unittest.TestCase):
    @staticmethod
    def make_instrument():
        from logos_instruments import HATS_DB_Functions

        instrument = object.__new__(HATS_DB_Functions)
        instrument.inst_id = "m4"
        instrument.CAL_RUN_TYPES = {7}
        instrument.MIN_CAL_INJECTIONS = 1
        instrument.norm = type(
            "Norm",
            (),
            {"run_type_column": "run_type_num", "standard_run_type": 8},
        )()
        instrument.db = CalibrationFakeDB()
        instrument.qurey_return_scale_num = lambda parameter_num: 110
        return instrument

    @staticmethod
    def make_row(**overrides):
        import pandas as pd

        row = {
            "run_time": pd.Timestamp("2024-07-15 08:51:00"),
            "tank_serial_num": "SX-3582",
            "run_type_num": 7,
            "rejected": 0,
            "mole_fraction": float("nan"),
            "analysis_num": 293241,
            "channel": "",
        }
        row.update(overrides)
        return row

    def test_null_only_group_deletes_stale_calibration(self):
        import pandas as pd

        instrument = self.make_instrument()
        df = pd.DataFrame([self.make_row()])

        instrument.upsert_calibrations(df, parameter_num=100)

        delete_calls = [
            (sql, params)
            for sql, params in instrument.db.query_calls
            if "DELETE FROM hats.calibrations" in sql
        ]
        self.assertEqual(len(delete_calls), 1)
        self.assertEqual(delete_calls[0][1][0], "SX-3582")
        self.assertEqual(delete_calls[0][1][3:], ("CH2I2", "M4", 100))
        self.assertEqual(instrument.db.multi_insert_calls, [])

    def test_group_with_valid_and_null_values_aggregates_valid_value(self):
        import pandas as pd

        instrument = self.make_instrument()
        df = pd.DataFrame(
            [
                self.make_row(),
                self.make_row(analysis_num=293242, mole_fraction=1.234),
            ]
        )

        instrument.upsert_calibrations(df, parameter_num=100)

        delete_calls = [
            sql
            for sql, _ in instrument.db.query_calls
            if "DELETE FROM hats.calibrations" in sql
        ]
        self.assertEqual(delete_calls, [])
        final_inserts = [
            call for call in instrument.db.multi_insert_calls if call[2]
        ]
        self.assertEqual(len(final_inserts), 1)
        _, params, all_rows = final_inserts[0]
        self.assertTrue(all_rows)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0][4:7], (1.234, 0.0, 1))
        self.assertEqual(params[0][10], 293242)

class CFC113aCalibrationTests(unittest.TestCase):
    def test_deconvolved_values_are_sent_to_matching_parameters(self):
        import pandas as pd
        from logos_instruments import M4_Instrument

        instrument = object.__new__(M4_Instrument)
        calls = []
        instrument.upsert_calibrations = lambda df, pnum: calls.append((pnum, df.copy()))
        source = pd.DataFrame(
            {
                "mole_fraction": [99.0],
                "mole_fraction_cfc113": [77.0],
                "mole_fraction_cfc113a": [0.12],
            }
        )

        instrument.upsert_cfc113a_calibrations(source)

        self.assertEqual([pnum for pnum, _ in calls], [32, 178])
        self.assertEqual(calls[0][1]["mole_fraction"].tolist(), [77.0])
        self.assertEqual(calls[1][1]["mole_fraction"].tolist(), [0.12])
        self.assertEqual(source["mole_fraction"].tolist(), [99.0])

if __name__ == '__main__':
    unittest.main()
