import contextlib
import datetime
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import m4_chemstation


class ReimportLockTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.gc = Path(temp.name)
        self.chemstation = self.gc / 'chemstation'
        self.run = self.chemstation / 'bd111926'
        self.chrom = self.gc / '26' / 'chromatograms' / 'channel0' / '261119.001'
        self.chrom.parent.mkdir(parents=True)
        self.chrom.write_text('old chromatogram')
        (self.gc / '.run-index').write_text('261119.001 old\n261118.001 keep\n')
        self.lock = io.StringIO()
        self.addCleanup(self.lock.close)
        for target, value in [('GC_DIR', self.gc), ('CHEMSTATION_DIR', self.chemstation)]:
            patcher = patch.object(m4_chemstation, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def acquire_after_ingest(self, name, log):
        self.assertEqual(name, 'm4')
        self.assertTrue(self.chrom.exists())
        # Simulate ingestion finishing the copy while reimport waits for its lock.
        self.run.mkdir(parents=True)
        return self.lock

    def test_waits_before_path_check_and_holds_lock_through_pipeline(self):
        def pipeline(gcd, yymm, run_date):
            self.assertFalse(self.lock.closed)
            self.assertEqual((gcd, yymm), (self.gc, '2611'))
            self.assertEqual(run_date.isoformat(), '2026-11-19')
            self.assertFalse(self.chrom.exists())
            self.assertEqual((self.gc / '.run-index').read_text(), '261118.001 keep\n')

        with patch('sys.argv', ['m4_chemstation.py', 'bd111926', '-y']), \
                patch.object(m4_chemstation.logos_lock, 'acquire',
                             side_effect=self.acquire_after_ingest) as acquire, \
                patch.object(m4_chemstation, 'run_commands', side_effect=pipeline) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            m4_chemstation.main()
        acquire.assert_called_once()
        run.assert_called_once()
        self.assertTrue(self.lock.closed)

    def test_aborting_releases_lock_without_removing_data(self):
        with patch('sys.argv', ['m4_chemstation.py', 'bd111926']), \
                patch.object(m4_chemstation.logos_lock, 'acquire',
                             side_effect=self.acquire_after_ingest), \
                patch('builtins.input', return_value='n'), \
                patch.object(m4_chemstation, 'run_commands') as run, \
                contextlib.redirect_stdout(io.StringIO()):
            m4_chemstation.main()
        self.assertTrue(self.lock.closed)
        self.assertTrue(self.chrom.exists())
        run.assert_not_called()

    def test_missing_directory_releases_lock_without_removing_data(self):
        with patch('sys.argv', ['m4_chemstation.py', 'bd111926', '-y']), \
                patch.object(m4_chemstation.logos_lock, 'acquire', return_value=self.lock), \
                patch.object(m4_chemstation, 'run_commands') as run, \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            m4_chemstation.main()
        self.assertEqual(error.exception.code, 2)
        self.assertTrue(self.lock.closed)
        self.assertTrue(self.chrom.exists())
        run.assert_not_called()


class ReimportPipelineTests(unittest.TestCase):
    def test_refreshes_sequence_serials_before_batching(self):
        with patch.object(m4_chemstation.subprocess, 'run') as run:
            m4_chemstation.run_commands(Path('/example/m4'), '2611', datetime.date(2026, 11, 19))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[2], [
            '/hats/gc/itxbin/m4_samplogs.py', '--all', '-i',
            '--refresh-serials', '2026-11-19',
        ])
        self.assertEqual(commands[-1][0], '/hats/gc/itxbin/m4_batch.py')
        self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))


if __name__ == '__main__':
    unittest.main()
