#!/usr/bin/env python3
"""Set mf_method_num for CATS air-port rows in ng_insitu_mole_fractions.

Applies CATS_Instrument.default_mf_method(pnum) to every air-port row from
--start-date onward: cal12 (method 2) for most analytes, cal1 (method 3)
for CCl4 (parameter_num=37). Pass --method to force a specific method
(ref/cal1/cal2/cal12) instead of each analyte's default.

Unlike ie3_set_mf_method.py, there is no baked-in date floor -- CATS has
decades of legacy history that was populated from published /aftp mole
fractions, not from a weekly cal12 fit. --start-date is required so a row
is only ever tagged cal12/cal1 in the same run that will actually recompute
its mole_fraction (see cats_batch.py --fits); tagging untouched legacy rows
would mislabel them as cal12-derived when they were never recomputed.

Before assigning cal12/cal1 to an analyte, checks that the required cal
tank(s) actually have a hats.scale_assignments entry for that parameter --
some analytes may have never been calibrated on the cal tanks, and blindly
tagging them cal12 leaves every mole_fraction NULL (no fit is ever
computable). Those analytes are left/reset to ref (method 1) instead, with a
printed note.

Safe to run repeatedly (idempotent) -- intended as a daily pipeline step
for newly-ingested rows (see cats_ingest.py).

Run with --dry-run to see row counts before committing.
"""

import argparse

from logos_instruments import CATS_Instrument


def _tank_serial(cats, port_num):
    if cats.port_config is None:
        return None
    mask = (
        (cats.port_config['site_num'] == cats.site_num)
        & (cats.port_config['port_num'] == port_num)
    )
    rows = cats.port_config.loc[mask, 'label']
    return rows.iat[0] if not rows.empty else None


def _has_scale_assignment(cats, tank, pnum):
    return tank is not None and cats.scale_assignments(tank, pnum) is not None


def resolve_methods(cats, method_override=None):
    """Return {pnum: method_num}, falling back to ref where the cal tank(s)
    needed for the desired method have no scale_assignments.

    method_override, if given, replaces default_mf_method(pnum) as the
    desired method for every analyte (e.g. force cal2 instead of the default
    cal12). Still validated per-method below -- cal2 needs the CAL2_PORT
    tank's scale_assignments, cal1 needs CAL1_PORT's, cal12 needs both.
    """
    cal1_tank = _tank_serial(cats, cats.CAL1_PORT)
    cal2_tank = _tank_serial(cats, cats.CAL2_PORT)

    rows = cats.db.doquery(
        f"SELECT DISTINCT param_num FROM hats.analyte_list "
        f"WHERE inst_num = {cats.inst_num}"
    )
    methods = {}
    for r in rows:
        pnum = int(r['param_num'])
        desired = method_override if method_override is not None else cats.default_mf_method(pnum)
        if desired == cats.MF_METHOD_CAL1:
            ok = _has_scale_assignment(cats, cal1_tank, pnum)
        elif desired == cats.MF_METHOD_CAL2:
            ok = _has_scale_assignment(cats, cal2_tank, pnum)
        elif desired == cats.MF_METHOD_CAL12:
            ok = (_has_scale_assignment(cats, cal1_tank, pnum)
                  and _has_scale_assignment(cats, cal2_tank, pnum))
        else:
            ok = True
        methods[pnum] = desired if ok else cats.MF_METHOD_REF
        if not ok:
            print(f"  NOTE: pnum={pnum} missing cal-tank scale_assignments; "
                  f"leaving on ref (method {cats.MF_METHOD_REF}) instead of "
                  f"{cats.MF_METHOD_LABELS[desired]}.")
    return methods


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--site', required=True,
                        help="CATS site code (e.g. brw, spo).")
    parser.add_argument('--start-date', required=True,
                        help="Only touch rows from this date onward "
                             "(YYYY-MM-DD). No default -- pass the same "
                             "window cats_batch.py will recompute.")
    parser.add_argument('--pnum', type=str, default=None,
                        help="Comma-separated parameter_num list to restrict to "
                             "(e.g. '5,6' for N2O,SF6 at CATS). Default: every "
                             "analyte for this instrument -- pass this to avoid "
                             "also reassigning unrelated analytes/channels.")
    parser.add_argument('--channel', type=str, default=None,
                        help="Restrict to this DB channel letter (e.g. 'q'). Default: "
                             "every channel this pnum reports on -- some analytes (e.g. "
                             "CATS N2O) are quantitated on more than one physical "
                             "channel, so omitting this can touch more than intended.")
    parser.add_argument('--method', type=str, default=None,
                        choices=['ref', 'cal1', 'cal2', 'cal12'],
                        help="Force this method instead of each analyte's default "
                             "(cal12, or cal1 for CCl4). Still falls back to ref if "
                             "the required cal tank(s) lack scale_assignments.")
    parser.add_argument('--dry-run', action='store_true',
                        help='Show counts only; do not update.')
    args = parser.parse_args()

    cats = CATS_Instrument(site=args.site)
    db = cats.db
    air_ports = cats.AIR_PORTS
    port_in = ', '.join(str(p) for p in air_ports)
    date_filter = f"AND a.analysis_time >= '{args.start_date}'"
    channel_filter = f"AND mf.channel = '{args.channel}'" if args.channel else ""

    method_override = None
    if args.method:
        name_to_num = {v: k for k, v in cats.MF_METHOD_LABELS.items()}
        method_override = name_to_num[args.method]
    methods = resolve_methods(cats, method_override=method_override)
    if args.pnum:
        wanted = {int(p) for p in args.pnum.split(',')}
        unknown = wanted - methods.keys()
        if unknown:
            parser.error(f"Unknown parameter_num(s) for {cats.inst_id} site {args.site}: "
                         f"{sorted(unknown)}")
        methods = {p: m for p, m in methods.items() if p in wanted}

    for pnum, method in sorted(methods.items()):
        count_sql = f"""
            SELECT COUNT(*) AS n
            FROM hats.ng_insitu_mole_fractions mf
            JOIN hats.ng_insitu_analysis a ON a.num = mf.analysis_num
            WHERE a.inst_num = {cats.inst_num}
              AND a.port IN ({port_in})
              AND mf.parameter_num = {pnum}
              {channel_filter}
              {date_filter}
        """
        n = db.doquery(count_sql)[0]['n']
        chan_label = args.channel or "all channels"
        print(f"  pnum={pnum} ({chan_label}) -> method {method} ({cats.MF_METHOD_LABELS[method]}): {n:,} rows")

        if args.dry_run or n == 0:
            continue

        update_sql = f"""
            UPDATE hats.ng_insitu_mole_fractions mf
            JOIN hats.ng_insitu_analysis a ON a.num = mf.analysis_num
            SET mf.mf_method_num = {method}
            WHERE a.inst_num = {cats.inst_num}
              AND a.port IN ({port_in})
              AND mf.parameter_num = {pnum}
              {channel_filter}
              {date_filter}
        """
        db.doquery(update_sql)

    if args.dry_run:
        print("--dry-run: no changes made.")
    else:
        print("Done.")


if __name__ == '__main__':
    main()
