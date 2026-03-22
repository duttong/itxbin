#!/usr/bin/env python3
"""Unified engineering data viewer — all instruments in one tabbed window."""

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from ie3eng import IE3EngWidget
from fe3eng import FE3EngWidget
from bld1eng import BLD1EngWidget


class EngPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Engineering Data Viewer')

        tabs = QTabWidget()
        tabs.addTab(IE3EngWidget(), 'IE3')
        tabs.addTab(FE3EngWidget(), 'FE3')
        tabs.addTab(BLD1EngWidget(), 'BLD1')

        self.setCentralWidget(tabs)
        self.resize(1400, 800)


def main():
    app = QApplication(sys.argv)
    win = EngPlotWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
