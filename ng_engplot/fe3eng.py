#!/usr/bin/env python3
"""FE3 Engineering Data Viewer."""

import gzip
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow

from engplot_base import EngPlotWidget

FE3_ROOT = Path('/hats/gc/fe3')
GSV_COLS = {'GSV1', 'GSV2', 'GSV3', 'GSV4', 'SSV0', 'SSV5'}


def scan_fe3_date_range() -> tuple[str, str] | None:
    """Return (YYYYMMDD_min, YYYYMMDD_max) from YYYYMMDD-HHMMSS directory names."""
    dates = sorted(
        d.name[:8] for d in FE3_ROOT.glob('??/incoming/????????-??????')
        if d.is_dir() and d.name[:8].isdigit()
    )
    return (dates[0], dates[-1]) if dates else None


def get_fe3_sample_columns() -> list[str]:
    """Read column names from the most recent eng CSV."""
    for yy_dir in sorted(FE3_ROOT.glob('??'), reverse=True):
        incoming = yy_dir / 'incoming'
        if not incoming.is_dir():
            continue
        for ts_dir in sorted(incoming.glob('????????-??????'), reverse=True):
            for f in sorted(ts_dir.glob('eng_[0-9]*.csv*'), reverse=True):
                try:
                    opener = gzip.open(f, 'rt') if f.name.endswith('.gz') else open(f, 'r')
                    with opener as fh:
                        return pd.read_csv(fh, nrows=0).columns.tolist()
                except Exception:
                    continue
    return []


def read_fe3_eng_file(path: Path) -> pd.DataFrame | None:
    try:
        opener = gzip.open(path, 'rt') if path.name.endswith('.gz') else open(path, 'r')
        with opener as fh:
            return pd.read_csv(fh, low_memory=False)
    except Exception as e:
        print(f'Warning: could not read {path}: {e}', file=sys.stderr)
        return None


class FE3EngWidget(EngPlotWidget):
    instrument_name = 'FE3'
    time_col = 'fe3_time'
    config_key = 'fe3'

    # ---------------------------------------------------------------- abstract impl

    def scan_date_range(self) -> tuple[str, str] | None:
        return scan_fe3_date_range()

    def get_columns(self) -> list[str]:
        return get_fe3_sample_columns()

    def load_data(self, end_date: date, n_days: int, resample: str) -> pd.DataFrame | None:
        start_date = end_date - timedelta(days=n_days - 1)
        frames = []

        for yy_dir in sorted(FE3_ROOT.glob('??')):
            if not yy_dir.is_dir():
                continue
            incoming = yy_dir / 'incoming'
            if not incoming.is_dir():
                continue
            for ts_dir in sorted(incoming.glob('????????-??????')):
                if not ts_dir.is_dir() or not ts_dir.name[:8].isdigit():
                    continue
                try:
                    ds = ts_dir.name[:8]
                    dir_date = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
                except ValueError:
                    continue
                if start_date <= dir_date <= end_date:
                    for f in sorted(ts_dir.glob('eng_[0-9]*.csv*')):
                        df = read_fe3_eng_file(f)
                        if df is not None:
                            frames.append(df)

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df['fe3_time'] = pd.to_datetime(df['fe3_time'], utc=True, errors='coerce').dt.tz_convert(None)
        df = df.dropna(subset=['fe3_time']).drop_duplicates('fe3_time').sort_values('fe3_time').set_index('fe3_time')

        for col in GSV_COLS:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes

        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if resample != '1s':
            df = df.resample(resample).mean()

        return df


class FE3EngWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FE3 Engineering Data Viewer')
        self.widget = FE3EngWidget()
        self.setCentralWidget(self.widget)
        self.resize(1300, 750)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = FE3EngWindow()
    win.show()
    sys.exit(app.exec_())
