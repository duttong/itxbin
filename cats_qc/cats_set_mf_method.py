#!/usr/bin/env python3
"""Set mf_method_num for CATS air-port rows in ng_insitu_mole_fractions.

Applies CATS_Instrument.default_mf_method(pnum) to every air-port row in the
--start/--end window (or from --start onward when --end is omitted): cal12
(method 2) for most analytes, cal1 (method 3)
for CCl4 (parameter_num=37). Pass --method to force a specific method
(ref/cal1/cal2/cal12) instead of each analyte's default.

Unlike ie3_set_mf_method.py, there is no baked-in date floor -- CATS has
decades of legacy history that was populated from published /aftp mole
fractions, not from a weekly cal12 fit. --start is required so a row
is only ever tagged cal12/cal1 in the same run that will actually recompute
its mole_fraction (see cats_batch.py --fits); tagging untouched legacy rows
would mislabel them as cal12-derived when they were never recomputed.

Before assigning cal12/cal1/cal2 to an analyte, checks -- separately for each
date sub-range in --start/--end -- that the cal tank(s) actually installed
during that sub-range have a hats.scale_assignments entry for that
parameter; blindly tagging a stretch cal12 with no assignment leaves every
mole_fraction in it NULL (no fit is ever computable). The check follows
cal-tank swaps within the window (via ng_port_info) and scale_assignments'
own fill boundaries, so a --start back to 1998 with a well-covered history
and only a recently-swapped, not-yet-measured current tank gets the
requested method for the covered years and falls back to ref (method 1),
with a printed note, only for the uncovered sub-range(s) -- it does not
reject the whole window over one bad tank at the tail.

Safe to run repeatedly (idempotent) -- intended as a daily pipeline step
for newly-ingested rows (see cats_ingest.py).

Run with --dry-run to see row counts before committing.

Week-boundary note: mf_method_num is stored per air-injection row, not per
week -- the "one method per analyte/channel/week" rule enforced elsewhere
(get_week_mf_method()'s modal lookup, set_week_mf_method()'s whole-week
write in the logos_data cal-week editor) is a convention, not a schema
constraint. --start/--end here are arbitrary calendar dates, deliberately
NOT snapped to week (W-SUN) boundaries -- useful for matching a tank swap
or scale correction's exact effective date -- so a --start/--end that lands
mid-week leaves that one week's rows genuinely split across two methods
until a later cats_set_mf_method.py call (or the GUI) re-covers the whole
week with one value. cats_batch.py's weekly cal12 fit (update_fits(),
called once per week per its OWN method resolution) and
_apply_week_methods_to_tanks() (which now only backfills tank rows, never
air rows -- see its docstring for the bug this used to cause) both handle
a split week correctly; nothing downstream re-derives or "fixes" a mixed
week's methods on its own.
"""

import argparse
from datetime import datetime

import pandas as pd

from logos_instruments import CATS_Instrument

# Sentinel standing in for "open-ended" (no upper bound yet) across the
# tank-occupancy and scale-assignment interval math below -- matches the
# convention already used for a currently-installed tank / a still-current
# fill (both have no real end date). Never emitted to SQL directly: any
# segment whose end lands here is printed/queried as an open upper bound
# instead (see _segment_date_filter).
_OPEN_END = pd.Timestamp('2099-01-01', tz='UTC')


def _parse_yyyymmdd(s):
    """Parse YYYYMMDD (or YYYY-MM-DD) into YYYY-MM-DD.

    This matches the date options used by the CATS QC scripts.
    """
    s = s.strip()
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date {s!r}; expected YYYYMMDD or YYYY-MM-DD."
    )


def _segment_date_filter(seg_start, seg_end):
    """Return the analysis-time SQL clause for one [seg_start, seg_end) segment.

    seg_start/seg_end are tz-aware pandas Timestamps (see _method_segments).
    Snapped to calendar dates -- this tool has always operated at day
    granularity (--start/--end are YYYYMMDD), so a tank swap or fill change
    that lands mid-day is attributed whole to the day it falls on, same
    coarseness as the rest of the tool.
    """
    clauses = [f"AND a.analysis_time >= '{seg_start.date()}'"]
    if seg_end < _OPEN_END:
        clauses.append(f"AND a.analysis_time < '{seg_end.date()}'")
    return '\n              '.join(clauses)


def _resolve_gas_channel(cats, gas_channel):
    """'CFC12_f' -> (pnum, channel), via the same display_name_ch lookup
    (cats.analytes, keyed 'DisplayName (channel)') the other cats_qc/
    scripts use (_resolve_pnum in cats_cal_method_qc.py/cats_cal_window_qc.py).
    Splits on the LAST underscore so multi-word/underscored gas names
    (none exist for CATS today, but matches sibling scripts' convention)
    aren't mis-split.
    """
    gas, channel = gas_channel.rsplit('_', 1)
    key = f"{gas} ({channel})".lower()
    lookup = {k.lower(): v for k, v in cats.analytes.items()}
    pnum = lookup.get(key)
    if pnum is None:
        raise ValueError(
            f"No analyte_list entry for {gas!r} ({channel!r}) at "
            f"{cats.inst_id} site {cats.site}"
        )
    return int(pnum), channel


def _all_pnums(cats):
    rows = cats.db.doquery(
        f"SELECT DISTINCT param_num FROM hats.analyte_list "
        f"WHERE inst_num = {cats.inst_num}"
    )
    return {int(r['param_num']) for r in rows}


def _port_occupancy_segments(cats, port_num, start_ts, end_ts):
    """Return [(tank_serial, seg_start, seg_end), ...] tiling [start_ts, end_ts)
    with no gaps, from cats.port_config_history. Assumes port_config_history
    covers the whole window (true for BRW/SPO back to their 1998 start); any
    leading stretch with no history row at all is silently skipped rather
    than raising, so a caller that hits this edge case sees no segment (and
    therefore no NOTE/UPDATE) for it instead of a crash.
    """
    history = cats.port_config_history
    if history is None or history.empty:
        return []
    sub = history.loc[
        (history['site_num'] == cats.site_num) & (history['port_num'] == int(port_num))
    ].dropna(subset=['start_datetime']).sort_values('start_datetime').reset_index(drop=True)
    if sub.empty:
        return []
    segs = []
    for i, row in sub.iterrows():
        seg_start = row['start_datetime']
        seg_end = sub['start_datetime'].iat[i + 1] if i + 1 < len(sub) else _OPEN_END
        clipped_start = max(seg_start, start_ts)
        clipped_end = min(seg_end, end_ts)
        if clipped_start < clipped_end:
            segs.append((row['label'], clipped_start, clipped_end))
    return segs


def _coverage_intervals(cats, port_num, pnum, start_ts, end_ts):
    """Tile [start_ts, end_ts) into (seg_start, seg_end, covered, tank):
    whether the tank installed on port_num at each moment has a
    hats.scale_assignments entry for pnum covering that moment. Follows both
    tank swaps (ng_port_info) and a tank's own refill/fill-code boundaries
    (scale_assignment_history), so a mid-occupancy refill that only some
    fills got measured for shows up as a real sub-range, not a blanket
    ok/gap per tank.
    """
    intervals = []
    for tank, occ_start, occ_end in _port_occupancy_segments(cats, port_num, start_ts, end_ts):
        cov = []
        for row in cats.scale_assignment_history(tank, pnum):
            a_start = row.get('start_date')
            if a_start is None:
                continue
            a_start = pd.Timestamp(a_start, tz='UTC')
            a_end_raw = row.get('end_date')
            a_end = pd.Timestamp(a_end_raw, tz='UTC') if a_end_raw else _OPEN_END
            cs, ce = max(a_start, occ_start), min(a_end, occ_end)
            if cs < ce:
                cov.append([cs, ce])
        cov.sort()
        merged = []
        for cs, ce in cov:
            if merged and cs <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], ce)
            else:
                merged.append([cs, ce])
        cursor = occ_start
        for cs, ce in merged:
            if cursor < cs:
                intervals.append((cursor, cs, False, tank))
            intervals.append((cs, ce, True, tank))
            cursor = ce
        if cursor < occ_end:
            intervals.append((cursor, occ_end, False, tank))
    return intervals


def _combine_and(cal1_intervals, cal2_intervals):
    """AND two (seg_start, seg_end, covered, tank) tilings of the same overall
    span into (seg_start, seg_end, covered, note) -- covered iff both the
    cal1 and cal2 tank in effect at that moment have coverage. note names
    whichever tank(s) are the problem when not covered.
    """
    breakpoints = sorted(
        {s for s, e, c, t in cal1_intervals} | {e for s, e, c, t in cal1_intervals}
        | {s for s, e, c, t in cal2_intervals} | {e for s, e, c, t in cal2_intervals}
    )

    def _at(intervals, ts):
        for s, e, c, t in intervals:
            if s <= ts < e:
                return c, t
        return False, None

    out = []
    for i in range(len(breakpoints) - 1):
        s, e = breakpoints[i], breakpoints[i + 1]
        c1, t1 = _at(cal1_intervals, s)
        c2, t2 = _at(cal2_intervals, s)
        covered = c1 and c2
        note = None
        if not covered:
            missing = []
            if not c1:
                missing.append(f"cal1 tank {t1}")
            if not c2:
                missing.append(f"cal2 tank {t2}")
            note = " and ".join(missing) + " missing scale_assignments"
        out.append((s, e, covered, note))
    return out


def _merge_segments(segments):
    """Merge adjacent (seg_start, seg_end, method, note) tuples that share a
    method into (seg_start, seg_end, method, notes) runs, collecting the
    distinct notes seen across the run (in order, de-duplicated)."""
    merged = []
    for s, e, method, note in segments:
        if merged and merged[-1][1] == s and merged[-1][2] == method:
            prev_s, prev_e, prev_method, prev_notes = merged[-1]
            notes = prev_notes if not note or note in prev_notes else prev_notes + [note]
            merged[-1] = (prev_s, e, prev_method, notes)
        else:
            merged.append((s, e, method, [note] if note else []))
    return merged


def _method_segments(cats, desired_method, pnum, start_ts, end_ts):
    """Return merged [(seg_start, seg_end, method_num, notes), ...] tiling
    [start_ts, end_ts): desired_method wherever the cal tank(s) it needs have
    scale_assignments coverage for pnum, cats.MF_METHOD_REF (with a note
    naming the uncovered tank) elsewhere. ref itself needs no cal tank, so it
    always covers the whole window untouched.
    """
    if desired_method == cats.MF_METHOD_REF:
        return [(start_ts, end_ts, cats.MF_METHOD_REF, [])]

    if desired_method == cats.MF_METHOD_CAL1:
        raw = [(s, e, c, (None if c else f"cal1 tank {t} has no scale_assignments"))
               for s, e, c, t in _coverage_intervals(cats, cats.CAL1_PORT, pnum, start_ts, end_ts)]
    elif desired_method == cats.MF_METHOD_CAL2:
        raw = [(s, e, c, (None if c else f"cal2 tank {t} has no scale_assignments"))
               for s, e, c, t in _coverage_intervals(cats, cats.CAL2_PORT, pnum, start_ts, end_ts)]
    elif desired_method == cats.MF_METHOD_CAL12:
        cal1 = _coverage_intervals(cats, cats.CAL1_PORT, pnum, start_ts, end_ts)
        cal2 = _coverage_intervals(cats, cats.CAL2_PORT, pnum, start_ts, end_ts)
        raw = _combine_and(cal1, cal2)
    else:
        raw = [(start_ts, end_ts, True, None)]

    segments = [
        (s, e, desired_method if covered else cats.MF_METHOD_REF, None if covered else note)
        for s, e, covered, note in raw
    ]
    return _merge_segments(segments)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--site', required=True,
                        help="CATS site code (e.g. brw, spo).")
    parser.add_argument('--start', dest='start_date', required=True,
                        type=_parse_yyyymmdd,
                        help="First date to touch (YYYYMMDD or YYYY-MM-DD). "
                             "Pass the same window cats_batch.py will recompute.")
    parser.add_argument('--end', type=_parse_yyyymmdd, default=None,
                        help="Last date to touch, inclusive (YYYYMMDD or YYYY-MM-DD). "
                             "Default: touch rows from --start onward.")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--pnum', type=str, default=None,
                        help="Comma-separated parameter_num list to restrict to "
                             "(e.g. '5,6' for N2O,SF6 at CATS). Default: every "
                             "analyte for this instrument -- pass this (or --gas) "
                             "to avoid also reassigning unrelated analytes/channels.")
    selector.add_argument('--gas', type=str, default=None,
                        help="Comma-separated Analyte_channel list to restrict to "
                             "(e.g. 'CFC12_f' or 'N2O_q,SF6_q'), same format as "
                             "cats_cal_method_qc.py/cats_cal_window_qc.py's --gas. "
                             "Alternative to --pnum; each entry carries its own "
                             "channel, so --channel is not used together with this.")
    parser.add_argument('--channel', type=str, default=None,
                        help="Restrict to this DB channel letter (e.g. 'q'). Default: "
                             "every channel this pnum reports on -- some analytes (e.g. "
                             "CATS N2O) are quantitated on more than one physical "
                             "channel, so omitting this can touch more than intended. "
                             "Not used together with --gas, which already carries a "
                             "channel per entry.")
    parser.add_argument('--method', type=str, default=None,
                        choices=['ref', 'cal1', 'cal2', 'cal12'],
                        help="Force this method instead of each analyte's default "
                             "(cal12, or cal1 for CCl4). Still falls back to ref, "
                             "per date sub-range, wherever the cal tank(s) installed "
                             "at that time lack scale_assignments coverage.")
    parser.add_argument('--dry-run', action='store_true',
                        help='Show counts only; do not update.')
    args = parser.parse_args()

    if args.end and args.end < args.start_date:
        parser.error('--end must not be before --start.')
    if args.gas and args.channel:
        parser.error('--channel is not used with --gas -- each --gas entry '
                     'already carries its own channel (e.g. CFC12_f).')

    cats = CATS_Instrument(site=args.site)
    db = cats.db
    air_ports = cats.AIR_PORTS
    port_in = ', '.join(str(p) for p in air_ports)
    start_ts = pd.Timestamp(args.start_date, tz='UTC')
    end_ts = (pd.Timestamp(args.end, tz='UTC') + pd.Timedelta(days=1)
              if args.end else _OPEN_END)

    method_override = None
    if args.method:
        name_to_num = {v: k for k, v in cats.MF_METHOD_LABELS.items()}
        method_override = name_to_num[args.method]

    all_pnums = _all_pnums(cats)

    # pnum_channel[pnum] is the channel to restrict that pnum's UPDATE/COUNT
    # to, or None for "every channel this pnum reports on". --gas resolves
    # each entry to its own (pnum, channel) pair; --pnum/default instead
    # share one global --channel value (possibly None) across every
    # selected pnum, matching the pre-existing single-channel-filter design.
    if args.gas:
        pnum_channel = {}
        for gas_channel in args.gas.split(','):
            try:
                pnum, channel = _resolve_gas_channel(cats, gas_channel.strip())
            except ValueError as exc:
                parser.error(str(exc))
            if pnum not in all_pnums:
                parser.error(f"Unknown parameter_num for {gas_channel!r} at "
                             f"{cats.inst_id} site {args.site}")
            pnum_channel[pnum] = channel
    else:
        if args.pnum:
            wanted = {int(p) for p in args.pnum.split(',')}
            unknown = wanted - all_pnums
            if unknown:
                parser.error(f"Unknown parameter_num(s) for {cats.inst_id} site {args.site}: "
                             f"{sorted(unknown)}")
        else:
            wanted = all_pnums
        pnum_channel = {p: args.channel for p in wanted}

    for pnum in sorted(pnum_channel):
        channel = pnum_channel[pnum]
        channel_filter = f"AND mf.channel = '{channel}'" if channel else ""
        chan_label = channel or "all channels"
        desired = method_override if method_override is not None else cats.default_mf_method(pnum)

        for seg_start, seg_end, method, notes in _method_segments(cats, desired, pnum, start_ts, end_ts):
            end_label = seg_end.date() if seg_end < _OPEN_END else 'now'
            for note in notes:
                print(f"  NOTE: pnum={pnum} ({chan_label}) {seg_start.date()} to {end_label}: "
                      f"{note}; using ref (method {cats.MF_METHOD_REF}) instead of "
                      f"{cats.MF_METHOD_LABELS[desired]}.")

            date_filter = _segment_date_filter(seg_start, seg_end)
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
            print(f"  pnum={pnum} ({chan_label}) {seg_start.date()} to {end_label} -> "
                  f"method {method} ({cats.MF_METHOD_LABELS[method]}): {n:,} rows")

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
