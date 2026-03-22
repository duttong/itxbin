#!/usr/bin/env python3
"""IE3 Engineering Data Viewer."""

import argparse
import gzip
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow,
)

from engplot_base import EngPlotWidget

SITE_ROOT = Path('/hats/gc')
VALID_SITES = ['smo', 'mlo', 'spo', 'brw']
GSV_COLS = {'GSV1', 'GSV2', 'GSV3'}


def load_header(site: str) -> list[str] | None:
    path = SITE_ROOT / site / 'eng_header.txt'
    if not path.exists():
        return None
    text = path.read_text()
    cols = [c.strip() for c in text.replace('\n', ',').split(',') if c.strip()]
    return cols or None


def read_eng_file(path: Path, cols: list[str]) -> pd.DataFrame | None:
    try:
        opener = gzip.open(path, 'rt') if path.name.endswith('.gz') else open(path, 'r')
        with opener as fh:
            return pd.read_csv(fh, header=None, names=cols, low_memory=False)
    except Exception as e:
        print(f'Warning: could not read {path}: {e}', file=sys.stderr)
        return None


def scan_ie3_date_range(site: str) -> tuple[str, str] | None:
    dirs = sorted(
        d.name for d in (SITE_ROOT / site).glob('??/incoming/????????')
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    )
    return (dirs[0], dirs[-1]) if dirs else None


class IE3EngWidget(EngPlotWidget):
    instrument_name = 'IE3'
    time_col = 'ie3_time'
    config_key = 'ie3'

    def __init__(self, default_site: str = 'smo', parent=None):
        self._default_site = default_site  # must be set before super().__init__()
        super().__init__(parent)

    # ---------------------------------------------------------------- hooks

    def _build_extra_controls(self, top: QHBoxLayout):
        top.addWidget(QLabel('Site:'))
        self.site_combo = QComboBox()
        self.site_combo.addItems(VALID_SITES)
        top.addWidget(self.site_combo)

    def _extra_restore_state(self, state: dict):
        site = state.get('last_site', self._default_site)
        idx = self.site_combo.findText(site)
        self.site_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _extra_save_state(self) -> dict:
        return {'last_site': self.site_combo.currentText()}

    def _connect_extra_signals(self):
        self.site_combo.currentTextChanged.connect(lambda _: self.setup_date_range())
        self.site_combo.currentTextChanged.connect(lambda _: self.save_state())

    # ---------------------------------------------------------------- abstract impl

    def scan_date_range(self) -> tuple[str, str] | None:
        return scan_ie3_date_range(self.site_combo.currentText())

    def get_columns(self) -> list[str]:
        return load_header(self.site_combo.currentText()) or []

    def load_data(self, end_date: date, n_days: int, resample: str) -> pd.DataFrame | None:
        site = self.site_combo.currentText()
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

        for col in GSV_COLS:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes

        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if resample != '1s':
            df = df.resample(resample).mean()

        return df


class IE3EngWindow(QMainWindow):
    def __init__(self, default_site: str = 'smo'):
        super().__init__()
        self.setWindowTitle('IE3 Engineering Data Viewer')
        self.widget = IE3EngWidget(default_site=default_site)
        self.setCentralWidget(self.widget)
        self.resize(1300, 750)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IE3 Engineering Data Viewer')
    parser.add_argument('--site', default='smo', choices=VALID_SITES,
                        help='Station code (default: smo)')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = IE3EngWindow(default_site=args.site)
    win.show()
    sys.exit(app.exec_())
