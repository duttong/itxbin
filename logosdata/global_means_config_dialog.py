"""Edit gml_global_means_config.yaml from the Timeseries tab.

``validate_config_text`` is the gate: text is only written when it parses and
every gas that has bins or a site override still builds a calculator, so a typo
cannot break the next Global Means export.
"""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPlainTextEdit, QPushButton, QVBoxLayout)

from global_means import CONFIG_FILE, GlobalMeansCalculator, GlobalMeansConfig


def validate_config_text(text: str) -> str | None:
    """Return an error message, or None when *text* is a usable config."""
    try:
        cfg = GlobalMeansConfig.from_text(text)
        # Build a calculator for each configured gas under both methods, so a
        # problem in gas_bins shows up here rather than at export time.
        gases = set(cfg.gas_bins) | set(cfg.gas_background_overrides)
        for method in ('bins', 'latitude'):
            cfg.weighting_method = method
            for gas in sorted(gases):
                if not cfg.sites_for(gas):
                    return f'{gas}: no background sites under weighting_method {method}'
                GlobalMeansCalculator(gas, config=cfg)
        if not cfg.header_template or not all(cfg.method_text.get(m) and cfg.columns_text.get(m)
                                              for m in ('latitude', 'bins')):
            return ('a header block is missing: global_means_file_header, '
                    'global_means_method_* and global_means_columns_* are all needed')
    except Exception as exc:  # yaml errors carry the line and column
        return str(exc)
    return None


class GlobalMeansConfigDialog(QDialog):
    """Plain-text editor for the global-means config, with validation on save."""

    def __init__(self, parent=None, path: Path = CONFIG_FILE):
        super().__init__(parent)
        self.path = Path(path)
        self.setWindowTitle('Global Means Config')
        self.resize(900, 700)

        self.editor = QPlainTextEdit()
        font = QFont('Monospace')
        font.setStyleHint(QFont.TypeWriter)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)

        info = QLabel(f'<b>{self.path}</b><br>'
                      'Saved edits apply to the next Global Means export. '
                      '<tt>deploy</tt> commits and pushes them.')
        info.setWordWrap(True)

        self.save_btn = QPushButton('Save')
        self.reload_btn = QPushButton('Reload from disk')
        self.cancel_btn = QPushButton('Close')
        self.save_btn.clicked.connect(self.save)
        self.reload_btn.clicked.connect(self.load)
        self.cancel_btn.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.reload_btn)
        buttons.addStretch()
        buttons.addWidget(self.cancel_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self.editor, stretch=1)
        layout.addLayout(buttons)
        self.load()

    def load(self) -> None:
        self.editor.setPlainText(self.path.read_text())
        self.editor.document().setModified(False)

    def save(self) -> bool:
        text = self.editor.toPlainText()
        if not text.endswith('\n'):
            text += '\n'
        error = validate_config_text(text)
        if error:
            QMessageBox.warning(self, 'Global Means Config',
                                f'Not saved -- the config is not valid:\n\n{error}')
            return False
        try:
            self.path.write_text(text)
        except OSError as exc:
            QMessageBox.warning(self, 'Global Means Config',
                                f'Could not write {self.path}:\n\n{exc}')
            return False
        self.editor.document().setModified(False)
        QMessageBox.information(self, 'Global Means Config', 'Saved.')
        return True

    def reject(self) -> None:
        if self.editor.document().isModified():
            answer = QMessageBox.question(
                self, 'Global Means Config', 'Discard unsaved changes?',
                QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
            if answer != QMessageBox.Discard:
                return
        super().reject()
