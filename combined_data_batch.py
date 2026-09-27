#!/usr/bin/env python3
"""Build the LOGOS combined data sets and store them in
hats.ng_logos_combined_data.

Each gas in logosdata/combined_data_config.yaml blends several measurement
programs (oldGC, RITS, fECD, CATS, IE3, CCGG, M*) into per-site monthly
means and semi-hemispheric, hemispheric and global means.  See that config for
the method and how it differs from the Igor Pro combined data.

The whole record of a gas is recomputed on every run and replaces the gas's
rows in the table.

Usage:
  python3 combined_data_batch.py                  # dry run, all gases
  python3 combined_data_batch.py CFC11 N2O        # dry run, two gases
  python3 combined_data_batch.py --compare        # also compare with /aftp
  python3 combined_data_batch.py --csv out_dir    # also write one CSV per gas
  python3 combined_data_batch.py -i               # write to the database
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'logosdata'))
sys.path.append('/ccg/src/db/')

from combined_data import CombinedConfig, CombinedDataBuilder  # noqa: E402
import db_utils.db_conn as db_conn  # noqa: E402

AFTP_ROOT = Path('/aftp/hats')
TABLE = 'hats.ng_logos_combined_data'


def read_published(gas_cfg: dict) -> pd.DataFrame | None:
    """The published Igor combined file for a gas, as a tidy frame with
    columns location, date, mean, sd.  None if there is no file."""
    folder = AFTP_ROOT / gas_cfg['gas_dir'] / 'combined'
    # SF6 has both an old GMD_ file and the current GML_ one: take the newest.
    matches = sorted((p for p in folder.glob(f"*_global_{gas_cfg['file_gas']}.txt")
                      if not p.is_symlink()), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    path = matches[-1]
    with open(path) as fh:
        lines = fh.readlines()
    header_idx = next(i for i, line in enumerate(lines) if not line.startswith('#'))
    cols = lines[header_idx].split()
    df = pd.read_csv(path, sep=r'\s+', skiprows=header_idx + 1, header=None, names=cols,
                     na_values=['nan'])
    prefix = cols[0].split('_')[0]
    gas = gas_cfg['file_gas']
    df['date'] = pd.to_datetime(dict(year=df[cols[0]], month=df[cols[1]], day=1))
    rows = []
    for col in cols[2:]:
        if col.endswith('_sd'):
            continue
        loc = col[len(prefix) + 1:-(len(gas) + 1)]
        part = df[['date', col]].rename(columns={col: 'mean'})
        part['sd'] = df.get(f'{col}_sd')
        part['location'] = {'Global': 'Global'}.get(loc, loc)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True).dropna(subset=['mean'])
    out.attrs['path'] = str(path)
    return out


def compare(result: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """Differences (new - published) per location over common months."""
    merged = result.merge(published, on=['location', 'date'], suffixes=('', '_pub'))
    merged['diff'] = merged['mean'] - merged['mean_pub']
    stats = merged.groupby('location').agg(
        months=('diff', 'size'),
        first=('date', 'min'), last=('date', 'max'),
        mean_diff=('diff', 'mean'), mean_abs=('diff', lambda d: d.abs().mean()),
        max_abs=('diff', lambda d: d.abs().max()),
        rel_pct=('diff', lambda d: np.nan),
    )
    rel = (merged['diff'].abs() / merged['mean_pub'].abs()).groupby(merged['location']).mean()
    stats['rel_pct'] = 100 * rel
    stats['first'] = stats['first'].dt.strftime('%Y-%m')
    stats['last'] = stats['last'].dt.strftime('%Y-%m')
    order = ['Global', 'NH', 'SH'] + sorted(set(stats.index) - {'Global', 'NH', 'SH'})
    return stats.reindex([o for o in order if o in stats.index])


def write_rows(db, gas: str, pnum: int, result: pd.DataFrame) -> int:
    db.doquery(f"DELETE FROM {TABLE} WHERE gas = %s", [gas], commit=True)
    params = [
        (gas, pnum, r.location, r.date.date(),
         None if pd.isna(r.mean) else float(r.mean),
         None if pd.isna(r.sd) else float(r.sd),
         int(r.n), r.programs)
        for r in result.itertuples(index=False)
    ]
    sql = (f"INSERT INTO {TABLE} (gas, parameter_num, location, month, mean, sd, n, programs) "
           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)")
    for i in range(0, len(params), 5000):
        db.doquery(sql, params[i:i + 5000], commit=True, multiInsert=True)
    return len(params)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('gases', nargs='*', help="Gas keys from the config; default all.")
    parser.add_argument('--compare', action='store_true',
                        help="Compare with the published /aftp combined files.")
    parser.add_argument('--csv', metavar='DIR', help="Write one CSV per gas to DIR.")
    parser.add_argument('-i', '--insert', action='store_true',
                        help=f"Replace each gas's rows in {TABLE}.")
    args = parser.parse_args()

    config = CombinedConfig.load()
    gases = args.gases or list(config.gases)
    unknown = set(gases) - set(config.gases)
    if unknown:
        parser.error(f"not in the config: {', '.join(sorted(unknown))}")

    db = db_conn.HATS_ng()
    pd.set_option('display.width', 200)
    for gas in gases:
        t0 = time.time()
        builder = CombinedDataBuilder(gas, db, config)
        result = builder.build()
        g = result[result.location == 'Global'].dropna(subset=['mean'])
        span = f"{g.date.min():%Y-%m} to {g.date.max():%Y-%m}" if not g.empty else 'no global mean'
        loaded = ', '.join(f"{p}:{len(df)}" for p, df in builder.program_data.items())
        print(f"{gas}: {len(result)} rows, Global {span} ({time.time() - t0:.1f}s)")
        print(f"  program site-months loaded: {loaded}")

        if args.csv:
            out = Path(args.csv)
            out.mkdir(parents=True, exist_ok=True)
            result.to_csv(out / f"combined_{gas}.csv", index=False, float_format='%.4f')
        if args.compare:
            published = read_published(config.gases[gas])
            if published is None:
                print("  no published file to compare")
            else:
                print(f"  vs {published.attrs['path']} (new - published):")
                print(compare(result, published).round(3).to_string())
        if args.insert:
            n = write_rows(db, gas, int(config.gases[gas]['parameter_num']), result)
            print(f"  wrote {n} rows to {TABLE}")


if __name__ == '__main__':
    main()
