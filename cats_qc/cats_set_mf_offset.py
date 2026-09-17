#!/usr/bin/env python3
"""Manage hats.ng_insitu_mf_offsets: manual, documented mole-fraction
corrections for a dated period where an instrument's response was known to
be off by a fixed amount for reasons no cal-tank/method choice can fix --
the tank(s) used were themselves fine, only the instrument's response
during that window was wrong (a mole-sieve trap swap, a detector/plumbing
repair, etc.).

Applied automatically, on top of whichever mf_method a row already went
through, by CATS_Instrument.calc_mole_fraction() / IE3_Instrument's shared
base (see _apply_mf_offsets() / _load_mf_offsets() in
logos_instruments_insitu.py -- IE3 gets this for free too, since CATS
inherits the offset machinery from it; there is no ie3_set_mf_offset.py
yet, but one would be a thin CLI wrapper the same way ie3_set_mf_method.py
is to this script's cats_set_mf_method.py sibling).

Offsets are cached once per instrument-instance at __init__ (self.mf_offsets),
matching port_config_history/scale_assignment_history's own staleness
tolerance -- add/remove here take effect the next time something
instantiates CATS_Instrument (e.g. the next cats_batch.py run), not inside
an already-running process. Since the correction is baked into the stored
mole_fraction (not applied live at query time), rerun the batch recompute
for the affected window after add/remove, e.g.:

    cats_batch.py --site sum --gas cfc12_f --start 20150726 -i --fits

Usage::

    # Add a correction, with a documented reason (required)
    python3 cats_set_mf_offset.py add --site sum --gas CFC12_f \\
        --start 20150726 --end 20160829 --offset 5.0 --operator gdutton \\
        --reason "Mole sieve trap swap + channel 2 troubleshooting on \\
2015-07-25 (sum2015-log JD206) shifted baseline ~5 ppt low until the \\
2016-08 MFC/power-supply repair (sum2016-log JD225-236)."

    # List current corrections for a site (optionally filtered to one gas)
    python3 cats_set_mf_offset.py list --site sum
    python3 cats_set_mf_offset.py list --site sum --gas CFC12_f

    # Remove one by its row id (from `list`)
    python3 cats_set_mf_offset.py remove --site sum --id 3
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logos_instruments import CATS_Instrument


def _parse_yyyymmdd(s):
    s = s.strip()
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date {s!r}; expected YYYYMMDD or YYYY-MM-DD."
    )


def _resolve_gas_channel(cats, gas_channel):
    """'CFC12_f' -> (pnum, channel). Same display_name_ch lookup convention
    as cats_set_mf_method.py / cats_cal_method_qc.py."""
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


def _add(args, cats):
    pnum, channel = _resolve_gas_channel(cats, args.gas)
    sql = """
        INSERT INTO hats.ng_insitu_mf_offsets
            (inst_num, parameter_num, channel, start_datetime, end_datetime,
             offset_value, offset_type, reason, operator)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cats.db.doquery(sql, [
        cats.inst_num, pnum, channel, args.start, args.end,
        args.offset, args.type, args.reason, args.operator,
    ])
    end_label = args.end or 'open-ended'
    print(f"Added: {cats.inst_id}/{cats.site} pnum={pnum} channel={channel} "
          f"{args.start} -> {end_label}: {args.type} {args.offset:+g}")
    print("Rerun the batch recompute for this window (e.g. cats_batch.py "
          "-i --fits) so the stored mole_fraction reflects the change.")


def _list(args, cats):
    pnum_filter = channel_filter = ""
    if args.gas:
        pnum, channel = _resolve_gas_channel(cats, args.gas)
        pnum_filter = f"AND parameter_num = {pnum}"
        channel_filter = f"AND (channel = '{channel}' OR channel IS NULL)"
    rows = cats.db.doquery(f"""
        SELECT num, parameter_num, channel, start_datetime, end_datetime,
               offset_value, offset_type, reason, operator, entry_date
        FROM hats.ng_insitu_mf_offsets
        WHERE inst_num = {cats.inst_num}
        {pnum_filter}
        {channel_filter}
        ORDER BY parameter_num, start_datetime
    """) or []
    if not rows:
        print(f"No offsets recorded for {cats.inst_id}/{cats.site}.")
        return
    for r in rows:
        pname = cats.analytes_inv.get(int(r['parameter_num']), f"pnum={r['parameter_num']}")
        chan_label = r['channel'] or 'all channels'
        end_label = r['end_datetime'] or 'open-ended'
        print(f"[{r['num']}] {pname}, channel={chan_label} "
              f"{r['start_datetime']} -> {end_label}: "
              f"{r['offset_type']} {r['offset_value']:+g}")
        print(f"      reason: {r['reason']}")
        print(f"      entered by {r['operator']} on {r['entry_date']}")


def _remove(args, cats):
    rows = cats.db.doquery(
        f"SELECT inst_num FROM hats.ng_insitu_mf_offsets WHERE num = {args.id}"
    )
    if not rows:
        print(f"No offset with id={args.id}.")
        return
    if int(rows[0]['inst_num']) != cats.inst_num:
        print(f"Offset [{args.id}] belongs to inst_num={rows[0]['inst_num']}, "
              f"not {cats.inst_id}/{cats.site} (inst_num={cats.inst_num}); "
              "refusing to remove -- pass the matching --site.")
        return
    cats.db.doquery(f"DELETE FROM hats.ng_insitu_mf_offsets WHERE num = {args.id}")
    print(f"Removed offset [{args.id}].")
    print("Rerun the batch recompute for the affected window (e.g. "
          "cats_batch.py -i --fits) so the stored mole_fraction reflects the change.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='action', required=True)

    p_add = sub.add_parser('add', help='Add a new correction period.')
    p_add.add_argument('--site', required=True, help="CATS site code (e.g. sum).")
    p_add.add_argument('--gas', required=True, help="Analyte_channel, e.g. CFC12_f.")
    p_add.add_argument('--start', required=True, type=_parse_yyyymmdd)
    p_add.add_argument('--end', default=None, type=_parse_yyyymmdd,
                        help="Omit for open-ended (still in effect).")
    p_add.add_argument('--offset', required=True, type=float,
                        help="Additive ppt/pptv amount, or multiplicative "
                             "factor (e.g. 1.01) if --type multiplicative.")
    p_add.add_argument('--type', dest='type', choices=['additive', 'multiplicative'],
                        default='additive')
    p_add.add_argument('--reason', required=True,
                        help="Why this correction exists -- required, stored "
                             "verbatim, and shown in the logos_data timeseries "
                             "tooltip on the offset's boundary lines.")
    p_add.add_argument('--operator', default=None)

    p_list = sub.add_parser('list', help='List recorded corrections.')
    p_list.add_argument('--site', required=True)
    p_list.add_argument('--gas', default=None, help="Restrict to one Analyte_channel.")

    p_remove = sub.add_parser('remove', help='Remove a correction by its row id.')
    p_remove.add_argument('--site', required=True,
                           help="Must match the offset's own site -- a safety "
                                "check against removing the wrong instrument's row.")
    p_remove.add_argument('--id', required=True, type=int)

    args = parser.parse_args()
    cats = CATS_Instrument(site=args.site)

    if args.action == 'add':
        _add(args, cats)
    elif args.action == 'list':
        _list(args, cats)
    elif args.action == 'remove':
        _remove(args, cats)


if __name__ == '__main__':
    main()
