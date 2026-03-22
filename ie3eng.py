#!/usr/bin/env python3
"""IE3 Engineering Data Viewer — PyQt5/Matplotlib timeseries display."""

import argparse
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDateEdit, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

SITE_ROOT = Path('/hats/gc')
CONFIG_PATH = Path.home() / '.ie3eng.json'
VALID_SITES = ['smo', 'mlo', 'spo', 'brw']
RESAMPLE_OPTIONS = ['1s', '10s', '1min', '5min', '10min']
GSV_COLS = {'GSV1', 'GSV2', 'GSV3'}
AUTO_RESAMPLE_DAYS = 3      # >= this many days → 1min, else 1s


def load_header(site: str) -> list[str] | None:
    path = SITE_ROOT / site / 'eng_header.txt'
    if not path.exists():
        return None
    text = path.read_text()
    cols = [c.strip() for c in text.replace('\n', ',').split(',') if c.strip()]
    return cols or None


def scan_date_range(site: str) -> tuple[str, str] | None:
    """Return (date_min, date_max) strings YYYYMMDD by scanning directory names."""
    dirs = sorted(
        d.name for d in (SITE_ROOT / site).glob('??/incoming/????????')
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
    return (dirs[0], dirs[-1]) if dirs else None


def scan_current_year_start(site: str) -> str | None:
    """Return the first YYYYMMDD directory in the current year's YY/incoming/."""
    yy = date.today().strftime('%y')
    dirs = sorted(
        d.name for d in (SITE_ROOT / site / yy / 'incoming').glob('????????')
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
    return dirs[0] if dirs else None


def read_eng_file(path: Path, cols: list[str]) -> pd.DataFrame | None:
    try:
        opener = gzip.open(path, 'rt') if path.name.endswith('.gz') else open(path, 'r')
        with opener as fh:
            return pd.read_csv(fh, header=None, names=cols, low_memory=False)
    except Exception as e:
        print(f'Warning: could not read {path}: {e}', file=sys.stderr)
        return None


def load_data(site: str, end_date: date, n_days: int, resample: str) -> pd.DataFrame | None:
    cols = load_header(site)
    if cols is None:
        print(f'No header file found for site {site}', file=sys.stderr)
        return None

    start_date = end_date - timedelta(days=n_days - 1)
    frames = []
    current = start_date
    while current <= end_date:
        day_dir = SITE_ROOT / site / current.strftime('%y') / 'incoming' / current.strftime('%Y%m%d')
        if day_dir.is_dir():
            for f in sorted(day_dir.glob('eng*.csv*')):
                df = read_eng_file(f, cols)
                if df is not None:
                    frames.append(df)
        current += timedelta(days=1)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df['ie3_time'] = pd.to_datetime(df['ie3_time'], utc=True, errors='coerce').dt.tz_convert(None)
    df = df.dropna(subset=['ie3_time']).drop_duplicates('ie3_time').sort_values('ie3_time').set_index('ie3_time')

    # Encode GSV string columns as integer category codes
    for col in GSV_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes  # -1=NaN, 0/1/2=categories

    # Coerce any remaining object columns to numeric
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if resample != '1s':
        df = df.resample(resample).mean()

    return df


class IE3EngWindow(QMainWindow):

    def __init__(self, default_site: str = 'smo'):
        super().__init__()
        self.setWindowTitle('IE3 Engineering Data Viewer')
        self.config = self._load_config()
        self._build_ui()
        self._restore_state(default_site)

    # ------------------------------------------------------------------ config

    def _load_config(self) -> dict:
        if CONFIG_PATH.exists():
            try:
                return json.loads(CONFIG_PATH.read_text())
            except Exception:
                pass
        return {}

    def _save_config(self):
        CONFIG_PATH.write_text(json.dumps(self.config, indent=2))

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top controls row
        top = QHBoxLayout()

        top.addWidget(QLabel('Site:'))
        self.site_combo = QComboBox()
        self.site_combo.addItems(VALID_SITES)
        self.site_combo.currentTextChanged.connect(self._on_site_changed)
        top.addWidget(self.site_combo)

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

        # Trace selector row
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

        # Matplotlib canvas + toolbar
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.resize(1300, 750)

    def _populate_trace_combos(self, columns: list[str]):
        left_cur = self.left_combo.currentText()
        right_cur = self.right_combo.currentText()

        plottable = [c for c in columns if c != 'ie3_time']

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

    # ------------------------------------------------------------------ slots

    def _go_to_most_recent(self):
        self.start_date.setDate(self.start_date.maximumDate())

    def _on_site_changed(self, site: str):
        dr = scan_date_range(site)
        if dr:
            date_min, date_max = dr
            qmin = QDate.fromString(date_min, 'yyyyMMdd')
            qmax = QDate.fromString(date_max, 'yyyyMMdd')
            self.start_date.setMinimumDate(qmin)
            self.start_date.setMaximumDate(qmax)
            self.start_date.setDate(qmax)
            if site not in self.config:
                self.config[site] = {}
            self.config[site]['date_min'] = date_min
            self.config[site]['date_max'] = date_max
            self._save_config()

        cols = load_header(site)
        if cols:
            self._populate_trace_combos(cols)

    def _on_days_changed(self, value: int):
        auto = '1min' if value >= AUTO_RESAMPLE_DAYS else '1s'
        idx = self.resample_combo.findText(auto)
        if idx >= 0:
            self.resample_combo.setCurrentIndex(idx)

    def _load_and_plot(self):
        site = self.site_combo.currentText()
        end_date = self.start_date.date().toPyDate()
        n_days = self.days_spin.value()
        resample = self.resample_combo.currentText()
        left_col = self.left_combo.currentText()
        right_col = self.right_combo.currentText()

        self.load_btn.setText('Loading…')
        self.load_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            df = load_data(site, end_date, n_days, resample)
        finally:
            self.load_btn.setText('Load')
            self.load_btn.setEnabled(True)

        if df is None or df.empty:
            print('No data found for selected range.', file=sys.stderr)
            return

        # Refresh combos with actual loaded columns, preserve selection
        self._populate_trace_combos(['ie3_time'] + list(df.columns))
        # Re-read selections after combo refresh
        left_col = self.left_combo.currentText()
        right_col = self.right_combo.currentText()

        self._plot(df, left_col, right_col, resample)

    # ------------------------------------------------------------------ plot

    def _y_limits_in_view(self, df: pd.DataFrame, col: str, xlim: tuple) -> tuple | None:
        """Return (ymin, ymax) for col within the current x view, with 5% margin."""
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
        # Preserve zoom/pan state across trace changes
        xlim = self.figure.axes[0].get_xlim() if self.figure.axes else None

        # Store state for xlim_changed callback
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
        ax1.set_title(f"{self.site_combo.currentText().upper()} — {', '.join(title_parts)}  [{resample}]")
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

    # ------------------------------------------------------------------ state

    def _save_ui_state(self):
        self.config.update({
            'last_site': self.site_combo.currentText(),
            'last_days': self.days_spin.value(),
            'last_resample': self.resample_combo.currentText(),
            'last_left_trace': self.left_combo.currentText(),
            'last_right_trace': self.right_combo.currentText(),
        })
        self._save_config()

    def _restore_state(self, default_site: str):
        site = self.config.get('last_site', default_site)
        idx = self.site_combo.findText(site)
        self.site_combo.setCurrentIndex(idx if idx >= 0 else 0)
        # Force site setup in case index didn't change (no signal fired)
        self._on_site_changed(self.site_combo.currentText())

        n_days = self.config.get('last_days', 5)
        self.days_spin.setValue(n_days)

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

        # Connect on-the-fly saving after restoring to avoid premature saves
        self.site_combo.currentTextChanged.connect(self._save_ui_state)
        self.days_spin.valueChanged.connect(self._save_ui_state)
        self.resample_combo.currentTextChanged.connect(self._save_ui_state)
        self.left_combo.currentTextChanged.connect(self._save_ui_state)
        self.right_combo.currentTextChanged.connect(self._save_ui_state)

        # Auto-load when trace selection changes
        self.left_combo.currentTextChanged.connect(self._load_and_plot)
        self.right_combo.currentTextChanged.connect(self._load_and_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IE3 Engineering Data Viewer')
    parser.add_argument('--site', default='smo', choices=VALID_SITES,
                        help='Station code (default: smo)')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = IE3EngWindow(default_site=args.site)
    win.show()
    sys.exit(app.exec_())
