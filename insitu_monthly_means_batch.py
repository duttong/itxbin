#!/usr/bin/env python3
"""Refresh hats.ng_insitu_monthly_means: monthly mean/std/n mole-fraction
rollups for IE3 and CATS, so logos_compare doesn't have to aggregate the
full in-situ history live on every plot.

Aggregation mirrors TimeseriesWidget.query_insitu_monthly_mean_data()
(logosdata/logos_timeseries.py): unrejected (rejected=0), air-port rows from
hats.ng_insitu_data_view, restricted to each analyte's preferred channel per
hats.ng_preferred_channel (date-aware -- falls back to including every
channel when no preference row exists for that inst/parameter, same as the
live GUI query), grouped by (site_num, parameter_num, calendar month).

Neither hats.ng_insitu_mole_fractions nor hats.ng_insitu_mole_fraction_tags
carries a modification timestamp, so a rejection applied today to 2015 data
is indistinguishable from one applied to this week's data -- there's no
cheap way to detect "what changed since last run." Each invocation therefore
recomputes an instrument's entire history and upserts it (ON DUPLICATE KEY
UPDATE); at a few dozen analytes x a few hundred months per instrument this
is cheap even though the source aggregation scans the full ng_insitu_* history.
A row is deleted only if its (site_num, parameter_num, month) no longer
appears in the freshly computed set -- e.g. every injection in a month has
since been rejected.

Meant to run on a periodic cron/cli cadence (e.g. weekly), not after every
ingest.

Usage:
  python3 insitu_monthly_means_batch.py ie3 -i
  python3 insitu_monthly_means_batch.py cats brw -i
  python3 insitu_monthly_means_batch.py cats brw nwr mlo -i
  python3 insitu_monthly_means_batch.py --all -i

Omit -i for a dry run (prints row counts, no DB writes).
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from logos_instruments import CATS_Instrument, IE3_Instrument

CATS_SITES = ['brw', 'sum', 'nwr', 'mlo', 'smo', 'spo']


def _instrument_label(inst) -> str:
    label = inst.inst_id.upper()
    if inst.inst_id == 'cats':
        label += f"-{inst.site.upper()}"
    return label


def _build_instruments(args) -> list:
    if args.all:
        return [IE3_Instrument()] + [CATS_Instrument(site=s) for s in CATS_SITES]
    if args.instrument == 'ie3':
        if args.sites:
            raise SystemExit("Site codes are only valid with 'cats'.")
        return [IE3_Instrument()]
    if args.instrument == 'cats':
        sites = args.sites or CATS_SITES
        return [CATS_Instrument(site=s) for s in sites]
    raise SystemExit("Specify 'ie3', 'cats [sites...]', or --all.")


def compute_monthly_means(inst) -> pd.DataFrame:
    """Aggregate hats.ng_insitu_data_view for one instrument into monthly
    mean/std/n per (site_num, parameter_num, month), across every analyte
    at once (the preferred-channel filter is a per-row correlated subquery,
    so one GROUP BY covers the whole instrument)."""
    ports = ','.join(str(p) for p in inst.AIR_PORTS)
    floor_clause = (
        f"AND v.analysis_time >= '{inst.DATA_START_DATE}'"
        if getattr(inst, 'DATA_START_DATE', None) else ''
    )
    sql = f"""
    SELECT v.site_num, v.parameter_num,
        DATE_FORMAT(v.analysis_time, '%Y-%m-01') AS month,
        AVG(v.mole_fraction)         AS mean,
        STDDEV_SAMP(v.mole_fraction) AS std,
        COUNT(v.mole_fraction)       AS n
    FROM hats.ng_insitu_data_view v
    WHERE v.inst_num = {inst.inst_num}
      AND v.port IN ({ports})
      AND v.rejected = 0
      {floor_clause}
      AND v.channel = COALESCE(
            (SELECT pc.channel FROM hats.ng_preferred_channel pc
             WHERE pc.inst_num = v.inst_num AND pc.parameter_num = v.parameter_num
               AND pc.start_date <= v.analysis_time
             ORDER BY pc.start_date DESC LIMIT 1),
            (SELECT pc.channel FROM hats.ng_preferred_channel pc
             WHERE pc.inst_num = v.inst_num AND pc.parameter_num = v.parameter_num
             ORDER BY pc.start_date ASC LIMIT 1),
            v.channel)
    GROUP BY v.site_num, v.parameter_num, month
    ORDER BY v.parameter_num, month;
    """
    df = pd.DataFrame(inst.db.doquery(sql))
    if not df.empty:
        df['month'] = pd.to_datetime(df['month']).dt.date
    return df


def refresh(inst, write: bool, verbose: bool) -> None:
    label = _instrument_label(inst)
    t0 = time.time()
    df = compute_monthly_means(inst)
    n_analytes = df['parameter_num'].nunique() if not df.empty else 0
    print(f"{label}: {len(df)} site-month-analyte rows across {n_analytes} "
          f"analytes (query {time.time() - t0:.1f}s)")
    if df.empty:
        return

    if not write:
        if verbose:
            print(df.head(10).to_string(index=False))
        print("  (dry run -- pass -i to write)")
        return

    upsert_sql = """
        INSERT INTO hats.ng_insitu_monthly_means
            (inst_num, site_num, parameter_num, month, mean, std, n)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            mean = VALUES(mean), std = VALUES(std), n = VALUES(n),
            mod_date = CURRENT_TIMESTAMP
    """
    params = [
        (
            inst.inst_num, int(r.site_num), int(r.parameter_num), r.month,
            float(r.mean) if pd.notna(r.mean) else None,
            float(r.std) if pd.notna(r.std) else None,
            int(r.n),
        )
        for r in df.itertuples(index=False)
    ]
    inst.db.doquery(upsert_sql, params, commit=True, multiInsert=True)
    print(f"  Upserted {len(params)} rows into hats.ng_insitu_monthly_means.")

    # An upsert never removes a row -- if every injection in a previously
    # published month has since been rejected, that (site, parameter, month)
    # drops out of the freshly computed set and must be deleted explicitly.
    fresh_keys = {
        (int(r.site_num), int(r.parameter_num), r.month)
        for r in df.itertuples(index=False)
    }
    existing = inst.db.doquery(
        "SELECT site_num, parameter_num, month FROM hats.ng_insitu_monthly_means "
        f"WHERE inst_num = {inst.inst_num}"
    ) or []
    stale = [
        (inst.inst_num, int(e['site_num']), int(e['parameter_num']), e['month'])
        for e in existing
        if (int(e['site_num']), int(e['parameter_num']), e['month']) not in fresh_keys
    ]
    if stale:
        inst.db.doquery(
            "DELETE FROM hats.ng_insitu_monthly_means "
            "WHERE inst_num = %s AND site_num = %s AND parameter_num = %s AND month = %s",
            stale, commit=True, multiInsert=True,
        )
        print(f"  Deleted {len(stale)} stale rows (no unrejected data left).")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('instrument', nargs='?', choices=['ie3', 'cats'],
                         help="Instrument to refresh.")
    parser.add_argument('sites', nargs='*', metavar='site',
                         help="CATS site codes (brw, sum, nwr, mlo, smo, spo); "
                              "omit to refresh all 6.")
    parser.add_argument('--all', action='store_true',
                         help="Refresh IE3 and every CATS site.")
    parser.add_argument('-i', '--insert', action='store_true',
                         help="Write results to DB (default: dry run).")
    parser.add_argument('-v', '--verbose', action='store_true',
                         help="Print a preview of computed rows on dry runs.")
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Specify 'ie3', 'cats [sites...]', or --all.")

    t0 = time.time()
    for inst in _build_instruments(args):
        refresh(inst, write=args.insert, verbose=args.verbose)
    print(f"\nDone. Elapsed {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
