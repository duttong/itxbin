#!/usr/bin/env python3
"""Set mf_method_num for IE3 air-port rows in ng_insitu_mole_fractions.

Applies IE3_Instrument.default_mf_method(pnum) to every air-port row from
--start-date onward: cal12 (method 2) for most analytes, cal1 (method 3)
for CCl4 (parameter_num=37). Rows before --start-date are left untouched --
cal12/cal1 need a real hats.ng_response weekly fit, which ie3_batch.py can
only build from CAL_MIN_DATE onward (IE3 didn't have both cal tanks
configured before then).

Before assigning cal12/cal1 to an analyte, checks that the required cal
tank(s) actually have a hats.scale_assignments entry for that parameter --
some analytes (e.g. CHCl3, pnum 34) have never been calibrated on the cal
tanks, and blindly tagging them cal12 leaves every mole_fraction NULL (no
fit is ever computable). Those analytes are left/reset to ref (method 1)
instead, with a printed note.

Safe to run repeatedly (idempotent) -- intended both as a one-time
historical backfill and as a daily pipeline step for newly-ingested rows
(see ie3_ingest.py).

Run with --dry-run to see row counts before committing.
"""

import argparse

from logos_instruments import IE3_Instrument

IE3_INST_NUM = 236
AIR_PORTS = (3, 7)

DEFAULT_START_DATE = "2026-03-01"  # matches ie3_batch.py CAL_MIN_DATE


def _tank_serial(ie3, port_num):
    if ie3.port_config is None:
        return None
    mask = (
        (ie3.port_config['site_num'] == ie3.site_num)
        & (ie3.port_config['port_num'] == port_num)
    )
    rows = ie3.port_config.loc[mask, 'label']
    return rows.iat[0] if not rows.empty else None


def _has_scale_assignment(ie3, tank, pnum):
    return tank is not None and ie3.scale_assignments(tank, pnum) is not None


def resolve_methods(ie3):
    """Return {pnum: method_num}, falling back to ref where the cal tank(s)
    needed for the analyte's default method have no scale_assignments."""
    cal1_tank = _tank_serial(ie3, ie3.CAL1_PORT)
    cal2_tank = _tank_serial(ie3, ie3.CAL2_PORT)

    rows = ie3.db.doquery(
        f"SELECT DISTINCT param_num FROM hats.analyte_list "
        f"WHERE inst_num = {IE3_INST_NUM}"
    )
    methods = {}
    for r in rows:
        pnum = int(r['param_num'])
        desired = ie3.default_mf_method(pnum)
        if desired == ie3.MF_METHOD_CAL1:
            ok = _has_scale_assignment(ie3, cal1_tank, pnum)
        elif desired == ie3.MF_METHOD_CAL12:
            ok = (_has_scale_assignment(ie3, cal1_tank, pnum)
                  and _has_scale_assignment(ie3, cal2_tank, pnum))
        else:
            ok = True
        methods[pnum] = desired if ok else ie3.MF_METHOD_REF
        if not ok:
            print(f"  NOTE: pnum={pnum} missing cal-tank scale_assignments; "
                  f"leaving on ref (method {ie3.MF_METHOD_REF}) instead of "
                  f"{ie3.MF_METHOD_LABELS[desired]}.")
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date', default=DEFAULT_START_DATE,
                        help=f"Only touch rows from this date onward "
                             f"(default: {DEFAULT_START_DATE}).")
    parser.add_argument('--dry-run', action='store_true',
                        help='Show counts only; do not update.')
    args = parser.parse_args()

    ie3 = IE3_Instrument()
    db = ie3.db
    port_in = ', '.join(str(p) for p in AIR_PORTS)
    date_filter = f"AND a.analysis_time >= '{args.start_date}'"

    methods = resolve_methods(ie3)

    for pnum, method in sorted(methods.items()):
        count_sql = f"""
            SELECT COUNT(*) AS n
            FROM hats.ng_insitu_mole_fractions mf
            JOIN hats.ng_insitu_analysis a ON a.num = mf.analysis_num
            WHERE a.inst_num = {IE3_INST_NUM}
              AND a.port IN ({port_in})
              AND mf.parameter_num = {pnum}
              {date_filter}
        """
        n = db.doquery(count_sql)[0]['n']
        print(f"  pnum={pnum} -> method {method} ({ie3.MF_METHOD_LABELS[method]}): {n:,} rows")

        if args.dry_run or n == 0:
            continue

        update_sql = f"""
            UPDATE hats.ng_insitu_mole_fractions mf
            JOIN hats.ng_insitu_analysis a ON a.num = mf.analysis_num
            SET mf.mf_method_num = {method}
            WHERE a.inst_num = {IE3_INST_NUM}
              AND a.port IN ({port_in})
              AND mf.parameter_num = {pnum}
              {date_filter}
        """
        db.doquery(update_sql)

    if args.dry_run:
        print("--dry-run: no changes made.")
    else:
        print("Done.")


if __name__ == '__main__':
    main()
