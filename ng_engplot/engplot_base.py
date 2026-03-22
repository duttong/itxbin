#!/usr/bin/env python3
"""Base widget for engineering data timeseries viewers."""

import json
import sys
from abc import abstractmethod
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDateEdit, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

RESAMPLE_OPTIONS = ['1s', '10s', '1min', '5min', '10min']


class _EngToolbar(NavigationToolbar2QT):
    """Navigation toolbar whose Home button autoscales to all loaded data."""

    def home(self, *args):
        for ax in self.canvas.figure.axes:
            ax.autoscale()
        self.canvas.draw_idle()
AUTO_RESAMPLE_DAYS = 3


class EngPlotWidget(QWidget):
    """Shared UI + plotting base for engineering data instruments.

    Subclasses must define class attributes:
        instrument_name : str   — shown in plot title
        time_col        : str   — datetime index column name (filtered from trace combos)
        config_path     : Path  — where to persist UI state as JSON

    Subclasses must implement:
        scan_date_range()  → tuple[str, str] | None   YYYYMMDD min/max
        get_columns()      → list[str]                for initial combo population
        load_data(end_date, n_days, resample) → pd.DataFrame | None

    Optional hooks for subclasses:
        _build_extra_controls(top)     add widgets before the date picker
        _extra_restore_state(state)    restore instrument-specific state (before setup_date_range)
        _extra_save_state()            return dict of extra keys to persist
        _connect_extra_signals()       connect instrument-specific signals after restore
    """

    instrument_name: str = 'Unknown'
    time_col: str = 'time'
    config_key: str = 'default'
    config_path: Path = Path.home() / '.engplot.json'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = self._load_config()
        self._df = None
        self._left_col = None
        self._right_col = None
        self._ax1 = None
        self._ax2 = None
        self._build_ui()
        self.restore_state()

    # ---------------------------------------------------------------- config

    def _load_config(self) -> dict:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text()).get(self.config_key, {})
            except Exception:
                pass
        return {}

    def _save_config(self):
        data = {}
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
            except Exception:
                pass
        data[self.config_key] = self.config
        self.config_path.write_text(json.dumps(data, indent=2))

    # ---------------------------------------------------------------- abstract

    @abstractmethod
    def scan_date_range(self) -> tuple[str, str] | None:
        """Return (YYYYMMDD_min, YYYYMMDD_max) by scanning data directories."""

    @abstractmethod
    def get_columns(self) -> list[str]:
        """Return column names for pre-populating trace combos on startup."""

    @abstractmethod
    def load_data(self, end_date: date, n_days: int, resample: str) -> pd.DataFrame | None:
        """Load, clean, and resample; return DataFrame indexed by time."""

    # ---------------------------------------------------------------- hooks

    def _build_extra_controls(self, top: QHBoxLayout):
        """Hook: add instrument-specific controls before the date picker."""

    def _extra_save_state(self) -> dict:
        """Hook: extra key/value pairs to merge into persisted state."""
        return {}

    def _extra_restore_state(self, state: dict):
        """Hook: restore instrument-specific state (called before setup_date_range)."""

    def _connect_extra_signals(self):
        """Hook: connect instrument-specific signals after state is restored."""

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self._build_extra_controls(top)

        most_recent_btn = QPushButton('Most Recent Data')
        most_recent_btn.clicked.connect(self._go_to_most_recent)
        top.addWidget(most_recent_btn)

        top.addWidget(QLabel('Start date:'))
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat('yyyy-MM-dd')
        top.addWidget(self.start_date)

        top.addWidget(QLabel('Last N days:'))
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 365)
        self.days_spin.setValue(5)
        self.days_spin.valueChanged.connect(self._on_days_changed)
        top.addWidget(self.days_spin)

        top.addWidget(QLabel('Resample:'))
        self.resample_combo = QComboBox()
        self.resample_combo.addItems(RESAMPLE_OPTIONS)
        top.addWidget(self.resample_combo)

        self.load_btn = QPushButton('Load')
        self.load_btn.clicked.connect(self._load_and_plot)
        top.addWidget(self.load_btn)
        top.addStretch()
        layout.addLayout(top)

        trace_row = QHBoxLayout()
        trace_row.addWidget(QLabel('Left trace:'))
        self.left_combo = QComboBox()
        self.left_combo.setMinimumWidth(160)
        trace_row.addWidget(self.left_combo)

        trace_row.addWidget(QLabel('Right trace:'))
        self.right_combo = QComboBox()
        self.right_combo.setMinimumWidth(160)
        trace_row.addWidget(self.right_combo)
        trace_row.addStretch()
        layout.addLayout(trace_row)

        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = _EngToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def _populate_trace_combos(self, columns: list[str]):
        left_cur = self.left_combo.currentText()
        right_cur = self.right_combo.currentText()

        plottable = [c for c in columns if c != self.time_col]

        self.left_combo.blockSignals(True)
        self.right_combo.blockSignals(True)

        self.left_combo.clear()
        self.right_combo.clear()
        self.right_combo.addItem('None')

        for col in plottable:
            self.left_combo.addItem(col)
            self.right_combo.addItem(col)

        li = self.left_combo.findText(left_cur)
        self.left_combo.setCurrentIndex(max(li, 0))
        ri = self.right_combo.findText(right_cur)
        self.right_combo.setCurrentIndex(max(ri, 0))

        self.left_combo.blockSignals(False)
        self.right_combo.blockSignals(False)

    # ---------------------------------------------------------------- slots

    def _go_to_most_recent(self):
        self.start_date.setDate(self.start_date.maximumDate())

    def setup_date_range(self):
        """Scan data directories to set date picker bounds; pre-populate combos."""
        dr = self.scan_date_range()
        if dr:
            date_min, date_max = dr
            self.start_date.setMinimumDate(QDate.fromString(date_min, 'yyyyMMdd'))
            self.start_date.setMaximumDate(QDate.fromString(date_max, 'yyyyMMdd'))
            self.start_date.setDate(QDate.fromString(date_max, 'yyyyMMdd'))
        cols = self.get_columns()
        if cols:
            self._populate_trace_combos(cols)

    def _on_days_changed(self, value: int):
        auto = '1min' if value >= AUTO_RESAMPLE_DAYS else '1s'
        idx = self.resample_combo.findText(auto)
        if idx >= 0:
            self.resample_combo.setCurrentIndex(idx)

    def _load_and_plot(self):
        end_date = self.start_date.date().toPyDate()
        n_days = self.days_spin.value()
        resample = self.resample_combo.currentText()

        self.load_btn.setText('Loading…')
        self.load_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            df = self.load_data(end_date, n_days, resample)
        finally:
            self.load_btn.setText('Load')
            self.load_btn.setEnabled(True)

        if df is None or df.empty:
            print('No data found for selected range.', file=sys.stderr)
            return

        self._populate_trace_combos([self.time_col] + list(df.columns))
        left_col = self.left_combo.currentText()
        right_col = self.right_combo.currentText()
        self._plot(df, left_col, right_col, resample)

    # ---------------------------------------------------------------- plot

    def _y_limits_in_view(self, df: pd.DataFrame, col: str, xlim: tuple) -> tuple | None:
        x0 = mdates.num2date(xlim[0]).replace(tzinfo=None)
        x1 = mdates.num2date(xlim[1]).replace(tzinfo=None)
        visible = df.loc[(df.index >= x0) & (df.index <= x1), col].dropna()
        if visible.empty:
            return None
        margin = (visible.max() - visible.min()) * 0.05 or abs(visible.mean()) * 0.05 or 0.1
        return (visible.min() - margin, visible.max() + margin)

    def _stats_label(self, col: str, xlim: tuple | None) -> str:
        df = self._df
        if xlim is not None:
            x0 = mdates.num2date(xlim[0]).replace(tzinfo=None)
            x1 = mdates.num2date(xlim[1]).replace(tzinfo=None)
            s = df.loc[(df.index >= x0) & (df.index <= x1), col].dropna()
        else:
            s = df[col].dropna()
        return f'{col}  {s.mean():.3f} ± {s.std():.3f}'

    def _on_xlim_changed(self, ax):
        xlim = ax.get_xlim()
        if self._left_col and self._left_col in self._df.columns:
            legend = self._ax1.get_legend()
            if legend and legend.texts:
                legend.texts[0].set_text(self._stats_label(self._left_col, xlim))
        if self._right_col and self._right_col != 'None' and self._right_col in self._df.columns:
            legend = self._ax2.get_legend()
            if legend and legend.texts:
                legend.texts[0].set_text(self._stats_label(self._right_col, xlim))
        self.canvas.draw_idle()

    def _plot(self, df: pd.DataFrame, left_col: str, right_col: str, resample: str):
        xlim = self.figure.axes[0].get_xlim() if self.figure.axes else None

        self._df = df
        self._left_col = left_col
        self._right_col = right_col
        self._ax2 = None

        self.figure.clear()
        ax1 = self.figure.add_subplot(111)
        ax1.set_xlabel('Time (UTC)')
        self._ax1 = ax1

        color_left = 'tab:blue'
        color_right = 'tab:red'
        has_left = left_col and left_col in df.columns
        has_right = right_col and right_col != 'None' and right_col in df.columns

        if has_left:
            ax1.plot(df.index, df[left_col], color=color_left, linewidth=0.8,
                     label=self._stats_label(left_col, xlim))
            ax1.set_ylabel(left_col, color=color_left)
            ax1.tick_params(axis='y', labelcolor=color_left)
            ax1.ticklabel_format(style='plain', axis='y', useOffset=False)
            ax1.legend(loc='upper left', fontsize=8)

        if has_right:
            ax2 = ax1.twinx()
            self._ax2 = ax2
            ax2.plot(df.index, df[right_col], color=color_right, linewidth=0.8,
                     label=self._stats_label(right_col, xlim))
            ax2.set_ylabel(right_col, color=color_right)
            ax2.tick_params(axis='y', labelcolor=color_right)
            ax2.ticklabel_format(style='plain', axis='y', useOffset=False)
            ax2.legend(loc='upper right', fontsize=8)

        title_parts = [p for p in [left_col, right_col if has_right else ''] if p]
        ax1.set_title(f"{self.instrument_name} — {', '.join(title_parts)}  [{resample}]")
        self.figure.autofmt_xdate()
        self.figure.tight_layout()

        if xlim is not None:
            ax1.set_xlim(xlim)
            if has_left:
                ylim = self._y_limits_in_view(df, left_col, xlim)
                if ylim:
                    ax1.set_ylim(ylim)
            if has_right:
                ylim = self._y_limits_in_view(df, right_col, xlim)
                if ylim:
                    ax2.set_ylim(ylim)

        ax1.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self.canvas.draw()

    # ---------------------------------------------------------------- state

    def save_state(self):
        self.config.update({
            'last_days': self.days_spin.value(),
            'last_resample': self.resample_combo.currentText(),
            'last_left_trace': self.left_combo.currentText(),
            'last_right_trace': self.right_combo.currentText(),
            **self._extra_save_state(),
        })
        self._save_config()

    def restore_state(self):
        self._extra_restore_state(self.config)
        self.setup_date_range()

        self.days_spin.setValue(self.config.get('last_days', 5))

        resample = self.config.get('last_resample', '')
        if resample:
            ri = self.resample_combo.findText(resample)
            if ri >= 0:
                self.resample_combo.setCurrentIndex(ri)

        left = self.config.get('last_left_trace', '')
        right = self.config.get('last_right_trace', 'None')
        if left:
            li = self.left_combo.findText(left)
            if li >= 0:
                self.left_combo.setCurrentIndex(li)
        if right:
            ri = self.right_combo.findText(right)
            if ri >= 0:
                self.right_combo.setCurrentIndex(ri)

        # Connect save-on-change after restoring to avoid premature saves
        self.days_spin.valueChanged.connect(lambda _: self.save_state())
        self.resample_combo.currentTextChanged.connect(lambda _: self.save_state())
        self.left_combo.currentTextChanged.connect(lambda _: self.save_state())
        self.right_combo.currentTextChanged.connect(lambda _: self.save_state())

        # Auto-reload when trace selection changes
        self.left_combo.currentTextChanged.connect(self._load_and_plot)
        self.right_combo.currentTextChanged.connect(self._load_and_plot)

        self._connect_extra_signals()
