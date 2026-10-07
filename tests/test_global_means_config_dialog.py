"""Validation and save behaviour of the Global Means config editor."""
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

from global_means import CONFIG_FILE  # noqa: E402
from global_means_config_dialog import (GlobalMeansConfigDialog,  # noqa: E402
                                        validate_config_text)

_app = QApplication.instance() or QApplication([])
SHIPPED = Path(CONFIG_FILE).read_text()


class ValidateTests(unittest.TestCase):
    def test_shipped_config_is_valid(self):
        self.assertIsNone(validate_config_text(SHIPPED))

    def test_bad_yaml(self):
        self.assertIsNotNone(validate_config_text(
            SHIPPED.replace('weighting_method: bins', 'weighting_method: [bins')))

    def test_bad_method(self):
        self.assertIn('weighting_method', validate_config_text(
            SHIPPED.replace('weighting_method: bins', 'weighting_method: sine')))

    def test_duplicate_bin_name_is_refused_not_silently_merged(self):
        dup = re.sub(r'SHmid:(\s+)\{sites: \[cgo\]', r'SHtrop:\1{sites: [cgo]',
                     SHIPPED, count=1)
        self.assertIn('duplicate key', validate_config_text(dup))

    def test_missing_header_block(self):
        self.assertIn('header', validate_config_text(
            SHIPPED.replace('global_means_method_bins:', 'x_method_bins:')))

    def test_not_a_mapping(self):
        self.assertIsNotNone(validate_config_text('- just\n- a list\n'))


class DialogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = Path(self.dir) / 'cfg.yaml'
        self.path.write_text(SHIPPED)
        patcher = mock.patch.multiple(QMessageBox, warning=mock.DEFAULT,
                                      information=mock.DEFAULT)
        self.boxes = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.dir)
        self.dlg = GlobalMeansConfigDialog(path=self.path)

    def test_opens_with_the_file(self):
        self.assertEqual(self.dlg.editor.toPlainText(), SHIPPED)

    def test_valid_edit_is_written(self):
        self.dlg.editor.setPlainText(SHIPPED.replace('phi: 30.0', 'phi: 25.0'))
        self.assertTrue(self.dlg.save())
        self.assertIn('phi: 25.0', self.path.read_text())

    def test_invalid_edit_leaves_the_file_alone(self):
        self.dlg.editor.setPlainText(SHIPPED.replace('weighting_method: bins',
                                                     'weighting_method: sine'))
        self.assertFalse(self.dlg.save())
        self.assertEqual(self.path.read_text(), SHIPPED)
        self.boxes['warning'].assert_called_once()

    def test_unwritable_target_is_reported(self):
        self.path.chmod(0o444)
        self.dlg.editor.setPlainText(SHIPPED.replace('phi: 30.0', 'phi: 25.0'))
        if os.access(self.path, os.W_OK):
            self.skipTest('running as a user who can write read-only files')
        self.assertFalse(self.dlg.save())
        self.boxes['warning'].assert_called_once()

    def test_reload_discards_edits(self):
        self.dlg.editor.setPlainText('junk')
        self.dlg.load()
        self.assertEqual(self.dlg.editor.toPlainText(), SHIPPED)


if __name__ == '__main__':
    unittest.main()
