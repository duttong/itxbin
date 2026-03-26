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


if __name__ == '__main__':
    unittest.main()
