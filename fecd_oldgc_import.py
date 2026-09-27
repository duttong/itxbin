#!/usr/bin/env python3
"""Load the original HATS flask GC-ECD ("oldGC", 1977-1995) monthly means
from the published /aftp files into hats.fecd_oldgc.

Only these monthly files survive: CFC-11, CFC-12 and N2O at ALT, BRW, CGO,
MLO, NWR, SMO and SPO.  Months with no samples (n = 0, mean nan) are skipped.
Rows are upserted on (site_num, parameter_num, month), so a rerun is safe.

Usage:
  python3 fecd_oldgc_import.py       # dry run
  python3 fecd_oldgc_import.py -i    # write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append('/ccg/src/db/')
import db_utils.db_conn as db_conn  # noqa: E402

AFTP_ROOT = Path('/aftp/hats')
OLDGC_INST_NUM = 251  # ccgg.inst_description: oldGC flask ECD, aka FE1

# file gas -> (directory under AFTP_ROOT, parameter_num)
OLDGC_GASES = {
    'F11': ('cfcs/cfc11/flasks/OldGC/monthly', 114),
    'F12': ('cfcs/cfc12/flasks/OldGC/monthly', 22),
    'N2O': ('n2o/flasks/OldGC/monthly', 5),
}


def read_file(path: Path) -> tuple[pd.DataFrame, str | None]:
    """(monthly rows with n > 0, calibration scale from the header)."""
    text = path.read_text()
    match = re.search(r'Calibration scale used:\s*(.+)', text)
    scale = match.group(1).strip() if match else None
    df = pd.read_csv(path, comment='#', sep=r'\s+', na_values=['nan', 'NaN'])
    df.columns = ['yr', 'mon', 'mean', 'sd', 'n']
    df = df[(df['n'] > 0) & df['mean'].notna()].copy()
    df['month'] = pd.to_datetime(dict(year=df.yr, month=df.mon, day=1)).dt.date
    return df[['month', 'mean', 'sd', 'n']], scale


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--insert', action='store_true',
                        help="Write to hats.fecd_oldgc (default: dry run).")
    args = parser.parse_args()

    db = db_conn.HATS_ng()
    site_nums = {r['code'].upper(): int(r['num'])
                 for r in db.doquery("SELECT num, code FROM gmd.site")}
    sql = """
        INSERT INTO hats.fecd_oldgc (site_num, inst_num, parameter_num, month, mean, sd, n, scale)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE inst_num = VALUES(inst_num), mean = VALUES(mean),
                                sd = VALUES(sd), n = VALUES(n), scale = VALUES(scale)
    """
    total = 0
    for gas, (folder, pnum) in OLDGC_GASES.items():
        for path in sorted((AFTP_ROOT / folder).glob(f'*_{gas}_MM.dat')):
            site = path.name.split('_')[0].upper()
            df, scale = read_file(path)
            print(f"{gas} {site}: {len(df)} months "
                  f"{df.month.min()} to {df.month.max()}, scale {scale}")
            total += len(df)
            if args.insert and not df.empty:
                params = [(site_nums[site], OLDGC_INST_NUM, pnum, r.month, float(r.mean),
                           None if pd.isna(r.sd) else float(r.sd), int(r.n), scale)
                          for r in df.itertuples(index=False)]
                db.doquery(sql, params, commit=True, multiInsert=True)
    print(f"{total} rows {'written' if args.insert else '(dry run -- pass -i to write)'}")


if __name__ == '__main__':
    main()
