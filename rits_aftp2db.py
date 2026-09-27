#!/usr/bin/env python3
"""Import RITS published hourly mole fractions from /aftp/hats into the
in-situ tables (hats.ng_insitu_analysis + hats.ng_insitu_mole_fractions).

RITS (Radiatively Important Trace Species) was the in-situ GC program that
preceded CATS: 1987 to 2001 at BRW, NWR, MLO, SMO and SPO, one instrument
per site (inst_num 246-250). Only the published, QC'd hourly values survive
(no chromatograms, ports, channels or calibrations), so each file row
becomes:

  ng_insitu_analysis        one row per sample time: run_time = analysis_time,
                            port = RITS_AIR_PORT (the single air inlet)
  ng_insitu_mole_fractions  one row per gas: mole_fraction = published value,
                            channel = '' (RITS has no channels),
                            mf_method_num = the 'published' method, so no
                            batch recalc treats these as recomputable.

The five gases at a site share one sample time per hour, so they land on the
same analysis row. The files carry no uncertainty, so unc stays NULL. All
values are on the current calibration scales.

Metadata is created idempotently on --insert: the 'published' row in
hats.ng_insitu_mf_methods, hats.analyte_list rows for each instrument/gas,
and one Air row per instrument in hats.ng_port_info.

Usage:
  python3 rits_aftp2db.py                 # dry run, all sites
  python3 rits_aftp2db.py brw mlo         # dry run, two sites
  python3 rits_aftp2db.py --insert        # write all sites
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append('/ccg/src/db/')
import db_utils.db_conn as db_conn  # noqa: E402

AFTP_ROOT = Path('/aftp/hats')

# site -> (inst_num, gmd.site num)
RITS_SITES = {
    'brw': (246, 15),
    'nwr': (247, 80),
    'mlo': (248, 75),
    'smo': (249, 112),
    'spo': (250, 113),
}

# file stem -> (path template, parameter_num, analyte_list display_name)
RITS_GASES = {
    'N2O':  ('n2o/insituGCs/RITS/hourly/{site}_N2O_All.dat', 5, 'N2O'),
    'F12':  ('cfcs/cfc12/insituGCs/RITS/hourly/{site}_F12_All.dat', 22, 'CFC12'),
    'F11':  ('cfcs/cfc11/insituGCs/RITS/hourly/{site}_F11_All.dat', 114, 'CFC11'),
    'CCl4': ('solvents/CCl4/insituGCs/RITS/hourly/{site}_CCl4_All.dat', 37, 'CCl4'),
    'MC':   ('solvents/CH3CCl3/insituGCs/RITS/hourly/{site}_MC_All.dat', 131, 'CH3CCl3'),
}

RITS_AIR_PORT = 1
RITS_CHANNEL = ''
PUBLISHED_METHOD = ('published', 'Published /aftp value (no recalc)')
AIR_PORT_TYPE = 1
BATCH = 5000


def read_rits_file(path: Path) -> pd.DataFrame:
    """Parse one RITS All.dat file into (analysis_time, mole_fraction)."""
    df = pd.read_csv(path, comment='#', sep=r'\s+', header=0,
                     na_values=['nan', 'NaN', 'Nan'])
    df.columns = ['yr', 'mon', 'day', 'hr', 'mn', 'mf']
    df['analysis_time'] = pd.to_datetime(
        dict(year=df.yr, month=df.mon, day=df.day, hour=df.hr, minute=df.mn))
    return df.dropna(subset=['mf'])[['analysis_time', 'mf']]


def read_site(site: str) -> dict[int, pd.DataFrame]:
    """{parameter_num: frame} for every RITS gas file present at a site."""
    out = {}
    for template, pnum, _ in RITS_GASES.values():
        path = AFTP_ROOT / template.format(site=site)
        if path.exists():
            out[pnum] = read_rits_file(path)
        else:
            print(f"  missing: {path}")
    return out


def published_method_num(db, write: bool) -> int | None:
    abbr, name = PUBLISHED_METHOD
    rows = db.doquery("SELECT num FROM hats.ng_insitu_mf_methods WHERE abbr = %s", [abbr])
    if rows:
        return int(rows[0]['num'])
    if not write:
        return None
    db.doquery("INSERT INTO hats.ng_insitu_mf_methods (abbr, name) VALUES (%s, %s)",
               [abbr, name], commit=True)
    return int(db.doquery("SELECT num FROM hats.ng_insitu_mf_methods WHERE abbr = %s",
                          [abbr])[0]['num'])


def ensure_metadata(db, site: str, first_time) -> None:
    """analyte_list rows and the ng_port_info Air row for one RITS site."""
    inst_num, site_num = RITS_SITES[site]
    for order, (_, pnum, display) in enumerate(RITS_GASES.values(), start=1):
        if not db.doquery("SELECT num FROM hats.analyte_list WHERE inst_num = %s AND param_num = %s",
                          [inst_num, str(pnum)]):
            db.doquery(
                "INSERT INTO hats.analyte_list (display_name, param_num, inst_num, start_date, "
                "channel, disp_order, public, agage, internal_use) "
                "VALUES (%s, %s, %s, %s, NULL, %s, 1, 0, 0)",
                [display, str(pnum), inst_num, '1987-01-01', order], commit=True)
    if not db.doquery("SELECT num FROM hats.ng_port_info WHERE inst_num = %s AND port_num = %s",
                      [inst_num, RITS_AIR_PORT]):
        db.doquery(
            "INSERT INTO hats.ng_port_info (site_num, inst_num, port_num, port_type_num, "
            "serial_number, start_datetime, comment) VALUES (%s, %s, %s, %s, 'air', %s, %s)",
            [site_num, inst_num, RITS_AIR_PORT, AIR_PORT_TYPE, first_time,
             'RITS single air inlet (published data only)'], commit=True)


def import_site(db, site: str, write: bool, method_num: int | None) -> None:
    inst_num, site_num = RITS_SITES[site]
    t0 = time.time()
    gases = read_site(site)
    if not gases:
        return
    times = sorted(set().union(*[set(df.analysis_time) for df in gases.values()]))
    n_mf = sum(len(df) for df in gases.values())
    print(f"{site.upper()} (inst {inst_num}): {len(times)} sample times, {n_mf} mole fractions, "
          f"{times[0]} to {times[-1]}")
    if not write:
        print("  (dry run -- pass --insert to write)")
        return

    ensure_metadata(db, site, times[0])

    analysis_sql = """
        INSERT INTO hats.ng_insitu_analysis
            (run_time, analysis_time, site_num, inst_num, port)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE run_time = VALUES(run_time), site_num = VALUES(site_num)
    """
    params = [(t, t, site_num, inst_num, RITS_AIR_PORT) for t in times]
    for i in range(0, len(params), BATCH):
        db.doquery(analysis_sql, params[i:i + BATCH], commit=True, multiInsert=True)

    rows = db.doquery(
        "SELECT num, analysis_time FROM hats.ng_insitu_analysis "
        "WHERE inst_num = %s AND port = %s", [inst_num, RITS_AIR_PORT])
    num_by_time = {pd.Timestamp(r['analysis_time']): int(r['num']) for r in rows}

    mf_sql = """
        INSERT INTO hats.ng_insitu_mole_fractions
            (analysis_num, parameter_num, channel, mole_fraction, mf_method_num)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE mole_fraction = VALUES(mole_fraction),
                                mf_method_num = VALUES(mf_method_num)
    """
    mf_params = []
    for pnum, df in gases.items():
        for t, mf in zip(df.analysis_time, df.mf):
            mf_params.append((num_by_time[pd.Timestamp(t)], pnum, RITS_CHANNEL,
                              round(float(mf), 4), method_num))
    for i in range(0, len(mf_params), BATCH):
        db.doquery(mf_sql, mf_params[i:i + BATCH], commit=True, multiInsert=True)
    print(f"  wrote {len(params)} analysis rows, {len(mf_params)} mole fractions "
          f"({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sites', nargs='*', metavar='site',
                        help="RITS sites (brw nwr mlo smo spo); default all.")
    parser.add_argument('-i', '--insert', action='store_true',
                        help="Write to the database (default: dry run).")
    args = parser.parse_args()
    unknown = set(args.sites) - set(RITS_SITES)
    if unknown:
        parser.error(f"unknown site(s): {', '.join(sorted(unknown))}")

    db = db_conn.HATS_ng()
    method_num = published_method_num(db, args.insert)
    for site in (args.sites or list(RITS_SITES)):
        import_site(db, site, args.insert, method_num)


if __name__ == '__main__':
    main()
