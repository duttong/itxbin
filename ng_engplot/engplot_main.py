#!/usr/bin/env python3
"""Unified engineering data viewer — all instruments in one tabbed window."""

import json
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from ie3eng import IE3EngWidget
from fe3eng import FE3EngWidget
from bld1eng import BLD1EngWidget

CONFIG_PATH = Path.home() / '.engplot.json'


class EngPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Engineering Data Viewer')

        self.tabs = QTabWidget()
        self.tabs.addTab(IE3EngWidget(), 'IE3')
        self.tabs.addTab(FE3EngWidget(), 'FE3')
        self.tabs.addTab(BLD1EngWidget(), 'BLD1')

        self.setCentralWidget(self.tabs)
        self.resize(1400, 800)

        self._restore_tab()
        self.tabs.currentChanged.connect(self._save_tab)

    def _restore_tab(self):
        try:
            data = json.loads(CONFIG_PATH.read_text())
            last = data.get('last_tab', '')
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == last:
                    self.tabs.setCurrentIndex(i)
                    break
        except Exception:
            pass

    def _save_tab(self, index: int):
        try:
            data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        except Exception:
            data = {}
        data['last_tab'] = self.tabs.tabText(index)
        CONFIG_PATH.write_text(json.dumps(data, indent=2))


def main():
    app = QApplication(sys.argv)
    win = EngPlotWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
