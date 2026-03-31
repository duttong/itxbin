#!/usr/bin/env python3
"""BLD1 Engineering Data Viewer."""

import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow

from engplot_base import EngPlotWidget

BLD1_ROOT = Path('/hats/gc/bld1')

# Filename pattern: YYYYbldDDDHHMM.NN.eng
_FNAME_RE = re.compile(r'^(\d{4})bld(\d{3})\d{4}\.\d+\.eng$')


def _parse_filename(path: Path) -> tuple[int, int] | None:
    """Return (year, doy) parsed from a bld1 eng filename, or None."""
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def scan_bld1_date_range() -> tuple[str, str] | None:
    """Return (YYYYMMDD_min, YYYYMMDD_max) from YYYYMMDD-HHMMSS directory names."""
    dates = sorted(
        d.name[:8] for d in BLD1_ROOT.glob('??/incoming/????????-??????')
        if d.is_dir() and d.name[:8].isdigit()
    )
    return (dates[0], dates[-1]) if dates else None


def get_bld1_sample_columns() -> list[str]:
    """Read column names from the most recent eng file (excluding Tsec*100)."""
    for yy_dir in sorted(BLD1_ROOT.glob('??'), reverse=True):
        incoming = yy_dir / 'incoming'
        if not incoming.is_dir():
            continue
        for ts_dir in sorted(incoming.glob('????????-??????'), reverse=True):
            for f in sorted(ts_dir.glob('*.eng'), reverse=True):
                if not _FNAME_RE.match(f.name):
                    continue
                try:
                    with open(f) as fh:
                        cols = pd.read_csv(fh, skiprows=2, nrows=0,
                                           sep=r'\s+', skipinitialspace=True,
                                           engine='python').columns.tolist()
                    return [c for c in cols if c != 'Tsec*100']
                except Exception:
                    continue
    return []


def read_bld1_eng_file(path: Path) -> pd.DataFrame | None:
    """Read one eng file, replacing Tsec*100 with a UTC bld1_time column."""
    parsed = _parse_filename(path)
    if parsed is None:
        return None
    year, doy = parsed
    base_date = date(year, 1, 1) + timedelta(days=doy - 1)

    # Local midnight in America/Denver as a Unix timestamp (handles DST)
    local_midnight = pd.Timestamp(
        year=base_date.year, month=base_date.month, day=base_date.day,
        tz='America/Denver'
    )
    local_midnight_unix = local_midnight.timestamp()

    try:
        with open(path) as fh:
            df = pd.read_csv(fh, skiprows=2, header=0,
                             sep=r'\s+', skipinitialspace=True,
                             engine='python')
    except Exception as e:
        print(f'Warning: could not read {path}: {e}', file=sys.stderr)
        return None

    if 'Tsec*100' not in df.columns:
        print(f'Warning: no Tsec*100 column in {path}', file=sys.stderr)
        return None

    tsec_sec = pd.to_numeric(df['Tsec*100'], errors='coerce') / 100.0
    utc_unix = local_midnight_unix + tsec_sec
    df['bld1_time'] = pd.to_datetime(utc_unix, unit='s', utc=True).dt.tz_convert(None)
    df = df.drop(columns=['Tsec*100'])
    return df


class BLD1EngWidget(EngPlotWidget):
    instrument_name = 'BLD1'
    time_col = 'bld1_time'
    config_key = 'bld1'

    # ---------------------------------------------------------------- abstract impl

    def scan_date_range(self) -> tuple[str, str] | None:
        return scan_bld1_date_range()

    def get_columns(self) -> list[str]:
        return get_bld1_sample_columns()

    def list_dirs_in_range(self, end_date: date, n_days: int) -> list[Path]:
        start_date = end_date - timedelta(days=n_days - 1)
        dirs = []
        for yy_dir in sorted(BLD1_ROOT.glob('??')):
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
                    dirs.append(ts_dir)
        return dirs

    def load_data(self, end_date: date, n_days: int, resample: str,
                  filter_dir: Path | None = None) -> pd.DataFrame | None:
        start_date = end_date - timedelta(days=n_days - 1)
        frames = []

        if filter_dir is not None:
            dirs_to_load = [filter_dir] if filter_dir.is_dir() else []
        else:
            dirs_to_load = []
            for yy_dir in sorted(BLD1_ROOT.glob('??')):
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
                        dirs_to_load.append(ts_dir)

        for ts_dir in dirs_to_load:
            for f in sorted(ts_dir.glob('*.eng')):
                df = read_bld1_eng_file(f)
                if df is not None:
                    frames.append(df)

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df = (df.dropna(subset=['bld1_time'])
                .drop_duplicates('bld1_time')
                .sort_values('bld1_time')
                .set_index('bld1_time'))

        for col in df.select_dtypes(include='object').columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if resample != '1s':
            df = df.resample(resample).mean()

        return df


class BLD1EngWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BLD1 Engineering Data Viewer')
        self.widget = BLD1EngWidget()
        self.setCentralWidget(self.widget)
        self.resize(1300, 750)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = BLD1EngWindow()
    win.show()
    sys.exit(app.exec_())
