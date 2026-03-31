#!/usr/bin/env python3
"""Unified engineering data viewer — all instruments in one tabbed window."""

import json
import sys
from pathlib import Path

import matplotlib as mpl
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
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
        self.tabs.currentChanged.connect(self._load_tab_if_needed)

        # Load only the initially selected tab
        QTimer.singleShot(0, lambda: self.tabs.currentWidget()._load_and_plot())

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

    def _load_tab_if_needed(self, index: int):
        widget = self.tabs.widget(index)
        if widget is not None and not widget._has_loaded:
            widget._load_and_plot()

    def _save_tab(self, index: int):
        try:
            data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        except Exception:
            data = {}
        data['last_tab'] = self.tabs.tabText(index)
        CONFIG_PATH.write_text(json.dumps(data, indent=2))


_STYLESHEET = """
QWidget {
    background-color: #F4F6F8;
    font-size: 11pt;
}
QLabel {
    background-color: transparent;
}
QPushButton {
    background-color: #4A90D9;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 4px 14px;
}
QPushButton:hover {
    background-color: #357ABD;
}
QPushButton:pressed {
    background-color: #2A5F9E;
}
QPushButton:disabled {
    background-color: #A0A0A0;
    color: #E0E0E0;
}
QPushButton:checked {
    background-color: #2A5F9E;
    color: white;
}
QComboBox, QDateEdit, QSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #C4CDD6;
    border-radius: 4px;
    padding: 2px 6px;
    min-height: 26px;
}
QSpinBox::up-button, QSpinBox::down-button {
    width: 16px;
    height: 13px;
}
QSpinBox::up-button {
    subcontrol-position: top right;
}
QSpinBox::down-button {
    subcontrol-position: bottom right;
}
QComboBox:focus, QDateEdit:focus, QSpinBox:focus {
    border-color: #4A90D9;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    selection-background-color: #D0E8FF;
    selection-color: #000000;
}
QTabWidget::pane {
    border: 1px solid #C4CDD6;
    background-color: #F4F6F8;
}
QTabBar::tab {
    background-color: #DDE3EA;
    border: 1px solid #C4CDD6;
    border-bottom: none;
    padding: 10px 28px;
    font-size: 12pt;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 3px;
}
QTabBar::tab:selected {
    background-color: #F4F6F8;
    color: #1A6BB5;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background-color: #C8D4E0;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(_STYLESHEET)
    app.setFont(QFont('Segoe UI, Arial, sans-serif', 11))

    # Matplotlib: match light background and scale fonts
    mpl.rcParams.update({
        'figure.facecolor': '#F4F6F8',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#C4CDD6',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'grid.color': '#E0E5EA',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
    })

    win = EngPlotWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
