#!/usr/bin/env python3
"""IE3 Engineering Data Viewer."""

import argparse
import gzip
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
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


SSV_ODD_PORTS = [1, 3, 5, 7, 9]

# site code (uppercase) → GML site_num
_SITE_NUMS = {'SMO': 112, 'MLO': 75, 'SPO': 113, 'BRW': 15}


def _get_port_names(site: str) -> dict[int, str]:
    """Return {port_num: serial_number} for odd ports at the given site."""
    site_num = _SITE_NUMS.get(site.upper())
    if site_num is None:
        return {}
    try:
        from logos_instruments import HATS_DB_Functions
        db = HATS_DB_Functions(inst_id='ie3')
        ports_str = ','.join(str(p) for p in SSV_ODD_PORTS)
        rows = db.doquery(
            f'SELECT port_num, serial_number FROM hats.ng_port_info '
            f'WHERE site_num = {site_num} AND port_num IN ({ports_str})'
        )
        return {r['port_num']: r['serial_number'] or f'Port {r["port_num"]}' for r in rows}
    except Exception as e:
        print(f'Warning: could not query port info: {e}', file=sys.stderr)
        return {}


class SampleFlowReportDialog(QDialog):
    def __init__(self, stats: list[dict], site: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Sample Flow Report — {site.upper()}')
        self.resize(520, 220)

        table = QTableWidget(len(stats), 6)
        table.setHorizontalHeaderLabels(['Port', 'Description', 'Mean', 'Std', 'Min', 'Max'])

        for i, row in enumerate(stats):
            table.setItem(i, 0, QTableWidgetItem(str(row['port'])))
            table.setItem(i, 1, QTableWidgetItem(str(row['description'])))
            for j, key in enumerate(('mean', 'std', 'min', 'max'), start=2):
                v = row[key]
                text = f'{v:.1f}' if v == v else 'nan'  # nan check
                item = QTableWidgetItem(text)
                table.setItem(i, j, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout = QVBoxLayout(self)
        layout.addWidget(table)


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

    def _build_right_controls(self, top: QHBoxLayout):
        self._report_btn = QPushButton('Sample Flow Report')
        self._report_btn.clicked.connect(self._show_sample_flow_report)
        top.addWidget(self._report_btn)

    def _connect_extra_signals(self):
        self.site_combo.currentTextChanged.connect(lambda _: self.setup_date_range())
        self.site_combo.currentTextChanged.connect(lambda _: self.save_state())

    def _show_sample_flow_report(self):
        if self._df is None:
            QMessageBox.information(self, 'No Data', 'Load data first.')
            return
        if 'flow_samp' not in self._df.columns or 'SSV0' not in self._df.columns:
            QMessageBox.warning(self, 'Missing Columns',
                                'flow_samp or SSV0 not found in loaded data.')
            return

        site = self.site_combo.currentText()
        port_names = _get_port_names(site)
        ssv0 = pd.to_numeric(self._df['SSV0'], errors='coerce').round()

        # Exclude first 5s after each SSV0 transition
        is_transition = (ssv0 != ssv0.shift(1)).fillna(True)
        last_transition = self._df.index.to_series().where(is_transition).ffill()
        steady = (self._df.index - last_transition).dt.total_seconds() >= 5

        stats = []
        for port in SSV_ODD_PORTS:
            s = self._df.loc[(ssv0 == port) & steady, 'flow_samp'].dropna()
            description = port_names.get(port, f'Port {port}')
            if s.empty:
                stats.append(dict(port=port, description=description,
                                  mean=float('nan'), std=float('nan'),
                                  min=float('nan'), max=float('nan')))
            else:
                stats.append(dict(port=port, description=description,
                                  mean=s.mean(), std=s.std(),
                                  min=s.min(), max=s.max()))

        dlg = SampleFlowReportDialog(stats, site, self)
        dlg.exec_()

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
