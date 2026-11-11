#!/usr/bin/env python3
"""Refresh hats.ng_insitu_monthly_means: monthly mean/std/n mole-fraction
rollups for IE3, CATS and RITS, so logos_compare doesn't have to aggregate
the full in-situ history live on every plot.

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
Use --gas to recompute the full history of a comma-separated list of gases.
A row is deleted only if its (site_num, parameter_num, month) no longer
appears in the freshly computed set -- e.g. every injection in a month has
since been rejected.

Meant to run on a periodic cron/cli cadence (e.g. weekly), not after every
ingest.

RITS (inst_num 246-250, 1987-2001) holds only published values imported by
rits_aftp2db.py and never changes, so it only needs one refresh after an
import and is left out of --all (the weekly cron). Its monthly std is the
scatter of the hourly values in the month -- the published files carry no
measurement uncertainty.

Usage:
  python3 insitu_monthly_means_batch.py ie3 -i
  python3 insitu_monthly_means_batch.py cats brw -i
  python3 insitu_monthly_means_batch.py cats brw nwr mlo -i
  python3 insitu_monthly_means_batch.py cats smo --gas cfc12,cfc113 -i
  python3 insitu_monthly_means_batch.py rits -i
  python3 insitu_monthly_means_batch.py --all -i

Omit -i for a dry run (prints row counts, no DB writes).
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from logos_instruments import CATS_Instrument, IE3_Instrument

CATS_SITES = ['brw', 'sum', 'nwr', 'mlo', 'smo', 'spo']

# RITS site -> inst_num; matches rits_aftp2db.RITS_SITES.
RITS_INST_NUM_BY_SITE = {'brw': 246, 'nwr': 247, 'mlo': 248, 'smo': 249, 'spo': 250}


class RITSSource:
    """The few attributes compute_monthly_means()/refresh() read from an
    instrument, for a RITS site. RITS has no instrument class: its rows are
    published values on a single air port with no channel."""

    inst_id = 'rits'
    AIR_PORTS = [1]  # rits_aftp2db.RITS_AIR_PORT
    DATA_START_DATE = None

    def __init__(self, site: str, db):
        self.site = site
        self.inst_num = RITS_INST_NUM_BY_SITE[site]
        self.db = db


def _instrument_label(inst) -> str:
    label = inst.inst_id.upper()
    if inst.inst_id in ('cats', 'rits'):
        label += f"-{inst.site.upper()}"
    return label


def _rits_sources(sites) -> list:
    db = IE3_Instrument().db
    return [RITSSource(s, db) for s in sites]


def _build_instruments(args) -> list:
    if args.all:
        # RITS is retired published data that never changes; refresh it
        # explicitly with 'rits' after a re-import instead.
        return [IE3_Instrument()] + [CATS_Instrument(site=s) for s in CATS_SITES]
    if args.instrument == 'ie3':
        if args.sites:
            raise SystemExit("Site codes are only valid with 'cats' or 'rits'.")
        return [IE3_Instrument()]
    if args.instrument == 'cats':
        sites = args.sites or CATS_SITES
        return [CATS_Instrument(site=s) for s in sites]
    if args.instrument == 'rits':
        sites = args.sites or list(RITS_INST_NUM_BY_SITE)
        unknown = set(sites) - set(RITS_INST_NUM_BY_SITE)
        if unknown:
            raise SystemExit(f"Unknown RITS site(s): {', '.join(sorted(unknown))}")
        return _rits_sources(sites)
    raise SystemExit("Specify 'ie3', 'cats [sites...]', 'rits [sites...]', or --all.")


def _resolve_gases(db, gas_list: str) -> tuple[int, ...]:
    """Resolve comma-separated HATS gas formulas, ignoring case and hyphens."""
    gases = dict.fromkeys(gas.strip().lower().replace('-', '') for gas in gas_list.split(','))
    if '' in gases:
        raise ValueError("Specify one or more gas formulas separated by commas, e.g. cfc12,cfc113.")
    placeholders = ', '.join(['%s'] * len(gases))
    rows = db.doquery(
        f"SELECT num, LOWER(REPLACE(formula, '-', '')) AS gas FROM gmd.parameter "
        f"WHERE LOWER(REPLACE(formula, '-', '')) IN ({placeholders})",
        tuple(gases),
    ) or []
    matches = {gas: set() for gas in gases}
    for row in rows:
        matches[row['gas']].add(int(row['num']))
    unknown = [gas for gas, ids in matches.items() if not ids]
    if unknown:
        raise ValueError("Unknown gas formula(s): " + ', '.join(unknown))
    ambiguous = [gas for gas, ids in matches.items() if len(ids) > 1]
    if ambiguous:
        raise ValueError("Ambiguous gas formula(s): " + ', '.join(ambiguous))
    return tuple(sorted(set.union(*matches.values())))


def _gas_filter(column: str, parameter_nums: tuple[int, ...]) -> str:
    if not parameter_nums:
        return ''
    ids = ', '.join(str(int(num)) for num in parameter_nums)
    return f"AND {column} IN ({ids})"


def compute_monthly_means(inst, parameter_nums: tuple[int, ...] = ()) -> pd.DataFrame:
    """Aggregate monthly air data per site and gas, using the preferred channel."""
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
      {_gas_filter('v.parameter_num', parameter_nums)}
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


def refresh(inst, write: bool, verbose: bool, parameter_nums: tuple[int, ...] = ()) -> None:
    label = _instrument_label(inst)
    t0 = time.time()
    df = compute_monthly_means(inst, parameter_nums)
    n_analytes = df['parameter_num'].nunique() if not df.empty else 0
    print(f"{label}: {len(df)} site-month-analyte rows across {n_analytes} "
          f"analytes (query {time.time() - t0:.1f}s)")
    if df.empty and not parameter_nums:
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
    if params:
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
        f"WHERE inst_num = {inst.inst_num} {_gas_filter('parameter_num', parameter_nums)}"
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
    parser.add_argument('instrument', nargs='?', choices=['ie3', 'cats', 'rits'],
                         help="Instrument to refresh.")
    parser.add_argument('sites', nargs='*', metavar='site',
                         help="CATS site codes (brw, sum, nwr, mlo, smo, spo) or RITS "
                              "site codes (brw, nwr, mlo, smo, spo); omit for all.")
    parser.add_argument('--all', action='store_true',
                         help="Refresh IE3 and every CATS site (not RITS).")
    parser.add_argument('--gas', metavar='GAS[,GAS...]',
                         help="Refresh only these gas formulas, e.g. cfc12,cfc113 (default: all gases).")
    parser.add_argument('-i', '--insert', action='store_true',
                         help="Write results to DB (default: dry run).")
    parser.add_argument('-v', '--verbose', action='store_true',
                         help="Print a preview of computed rows on dry runs.")
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Specify 'ie3', 'cats [sites...]', 'rits [sites...]', or --all.")

    t0 = time.time()
    instruments = _build_instruments(args)
    try:
        parameter_nums = _resolve_gases(instruments[0].db, args.gas) if args.gas is not None else ()
    except ValueError as error:
        parser.error(str(error))
    for inst in instruments:
        refresh(inst, write=args.insert, verbose=args.verbose, parameter_nums=parameter_nums)
    print(f"\nDone. Elapsed {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
