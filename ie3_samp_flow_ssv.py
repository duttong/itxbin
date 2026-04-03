#!/usr/bin/env python3
"""
For each time the IE3 SSV transitions from an odd (active) to an even (stop)
position, sample flow_samp 5 seconds before the switch.  Plot those points
vs. datetime, colored by the odd SSV port.

Nominal flow_samp is ~50 cc/min; deviations indicate a problem port/period.

Usage:
    python3 ie3_samp_flow_ssv.py [--site smo|mlo|spo|brw] [--days 7] [--out fig.png]
"""

import argparse
import gzip
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SITE_ROOT = Path('/hats/gc')
GSV_COLS = {'GSV1', 'GSV2', 'GSV3'}
LEGACY_HEADER_NAME = 'eng_header_smo_early.txt'

_CMAP = plt.get_cmap('tab10')
# Fixed color per odd SSV port (1,3,5,7,9)
COLOR_MAP = {1: _CMAP(0), 3: _CMAP(1), 5: _CMAP(2), 7: _CMAP(3), 9: _CMAP(4)}


# ─── data loading ────────────────────────────────────────────────────────────

def load_legacy_header(site: str) -> list[str] | None:
    path = SITE_ROOT / site / LEGACY_HEADER_NAME
    if not path.exists():
        return None
    cols = [c.strip() for c in path.read_text().replace('\n', ',').split(',') if c.strip()]
    return cols or None


def _read_eng_file(path: Path, fallback_cols: list[str] | None) -> pd.DataFrame | None:
    try:
        opener = gzip.open(path, 'rt') if path.name.endswith('.gz') else open(path, 'r')
        with opener as fh:
            raw = fh.read()
        first_line = raw[:raw.index('\n')] if '\n' in raw else raw
        if first_line.startswith('ie3_time'):
            return pd.read_csv(io.StringIO(raw), header=0, low_memory=False)
        elif fallback_cols:
            return pd.read_csv(io.StringIO(raw), header=None, names=fallback_cols,
                               low_memory=False)
        else:
            print(f'Warning: no header in {path} and no fallback cols available',
                  file=sys.stderr)
            return None
    except Exception as e:
        print(f'Warning: could not read {path}: {e}', file=sys.stderr)
        return None


def load_data(site: str, end_date: date, n_days: int) -> pd.DataFrame | None:
    fallback_cols = load_legacy_header(site)

    frames = []
    current = end_date - timedelta(days=n_days - 1)
    while current <= end_date:
        day_dir = (SITE_ROOT / site
                   / current.strftime('%y') / 'incoming' / current.strftime('%Y%m%d'))
        if day_dir.is_dir():
            for f in sorted(day_dir.glob('eng*.csv*')):
                df = _read_eng_file(f, fallback_cols)
                if df is not None:
                    frames.append(df)
        current += timedelta(days=1)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df['ie3_time'] = (pd.to_datetime(df['ie3_time'], format='mixed', utc=True, errors='coerce')
                        .dt.tz_convert(None))
    df = (df.dropna(subset=['ie3_time'])
            .drop_duplicates('ie3_time')
            .sort_values('ie3_time')
            .set_index('ie3_time'))

    for col in GSV_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ─── extract pre-transition samples ─────────────────────────────────────────

def extract_pretransition_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per odd→even SSV transition: mean flow_samp over -10 to -5 s."""
    ssv0 = pd.to_numeric(df['SSV0'], errors='coerce').round()

    # Identify rows where SSV0 just switched to an even value
    switched_to_even = ssv0.notna() & (ssv0 % 2 == 0) & (ssv0 != ssv0.shift(1))
    # The port we're leaving is the previous odd value
    prev_ssv = ssv0.shift(1)
    odd_to_even = switched_to_even & prev_ssv.notna() & (prev_ssv % 2 == 1)

    transition_times = df.index[odd_to_even]

    records = []
    for t in transition_times:
        window = df['flow_samp'].loc[t - pd.Timedelta(seconds=10) : t - pd.Timedelta(seconds=5)]
        if window.empty:
            continue
        port = int(prev_ssv.loc[t])
        records.append({'transition_time': t, 'port': port,
                        'flow_samp': window.mean(), 'n': len(window)})

    return pd.DataFrame(records)


# ─── plot ─────────────────────────────────────────────────────────────────────

def plot(pts: pd.DataFrame, site: str, n_days: int, out: str | None):
    fig, ax = plt.subplots(figsize=(14, 5))

    for port, grp in pts.groupby('port'):
        color = COLOR_MAP.get(port, 'gray')
        ax.scatter(grp['transition_time'], grp['flow_samp'],
                   color=color, label=f'SSV port {port}',
                   s=30, zorder=3, alpha=0.85)

    ax.axhline(50, color='black', linewidth=0.8, linestyle='--', label='nominal 50 cc/min')

    ax.set_xlabel('SSV transition time (UTC)')
    ax.set_ylabel('flow_samp 5 s before switch (cc/min)')
    end_dt = pts['transition_time'].max()
    start_dt = pts['transition_time'].min()
    ax.set_title(
        f'IE3 {site.upper()} — flow_samp 5 s before odd→even SSV switch\n'
        f'{start_dt.date()} – {end_dt.date()}'
    )
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150)
        print(f'Saved figure to {out}')
    else:
        plt.show()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IE3 flow_samp 5 s before each odd→even SSV transition')
    parser.add_argument('--site', default='smo', choices=['smo', 'mlo', 'spo', 'brw'])
    parser.add_argument('--days', type=int, default=7, metavar='N')
    parser.add_argument('--out', default=None, metavar='FILE',
                        help='Save figure to FILE instead of displaying')
    args = parser.parse_args()

    end_date = date.today()
    print(f'Loading {args.days} days of IE3 eng data for {args.site.upper()} …')
    df = load_data(args.site, end_date, args.days)

    if df is None:
        print('No data found.', file=sys.stderr)
        sys.exit(1)

    missing = [c for c in ('flow_samp', 'SSV0') if c not in df.columns]
    if missing:
        print(f'Columns not found in data: {missing}', file=sys.stderr)
        sys.exit(1)

    print(f'Loaded {len(df):,} rows  ({df.index.min()} – {df.index.max()})')

    pts = extract_pretransition_flow(df)
    if pts.empty:
        print('No odd→even SSV transitions found.', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(pts)} transitions across ports: '
          + ', '.join(f'port {p}: {n}' for p, n in pts.groupby('port').size().items()))

    plot(pts, args.site, args.days, args.out)


if __name__ == '__main__':
    main()
