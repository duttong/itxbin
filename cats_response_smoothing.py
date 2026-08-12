#!/usr/bin/env python3
"""Set detrend_method_num (response smoothing method) for CATS rows.

Sets the response-smoothing method used to normalize a CATS analyte's
response (see Normalizing.calculate_smoothed_std in logos_instruments_core.py)
across a date range, recomputes normalized_resp/mole_fraction with the new
method, and upserts hats.ng_insitu_mole_fractions -- the same write path
logos_data's GUI "Response smoothing" combobox uses, but applied in bulk
instead of one run_time at a time.

This only changes *how the reference-port response curve is smoothed*. It
does not touch mf_method_num (ref/cal12/cal1/cal2) or re-fit calibrations --
use --recalc (which shells out to cats_batch.py --fits -i) if the smoother
response should also feed into fresh weekly cal fits.

Usage examples:

  # Preview affected rows (dry run) for CATS-SPO N2O, channel q,
  # from 2026-03-01 to now
  python3 cats_response_smoothing.py --method 3 --site spo --analyte N2O \\
      --channel q --start 20260301

  # Apply point-to-point moving average and write to DB
  python3 cats_response_smoothing.py --method 3 --site spo --analyte N2O \\
      --channel q --start 20260301 -i

  # Apply and also recompute weekly fits + mole fractions via cats_batch.py
  python3 cats_response_smoothing.py --method 3 --site spo --analyte N2O \\
      --channel q --start 20260301 -i --recalc
"""

import argparse
import subprocess
import sys
from datetime import datetime

from logos_instruments import CATS_Instrument

METHOD_LABELS = {
    1: "point-to-point linear interpolation",
    3: "2-point moving average",
    2: "LOWESS, ~5 points",
    4: "3-point boxcar mean",
    6: "5-point boxcar mean",
    5: "LOWESS, ~10 points",
}


def _method_help() -> str:
    lines = ["detrend_method_num values (least -> most smoothing):"]
    for num, desc in METHOD_LABELS.items():
        lines.append(f"  {num}: {desc}")
    return "\n".join(lines)


def _parse_yyyymmdd(s: str) -> str:
    """Parse YYYYMMDD (or YYYY-MM-DD) into 'YYYY-MM-DD'."""
    s = s.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date {s!r}; expected YYYYMMDD or YYYY-MM-DD."
    )


def resolve_analyte(cats: CATS_Instrument, analyte: str, channel: str | None):
    """Resolve an analyte display name (+ optional channel) to (pnum, channel).

    Raises SystemExit with a helpful message if the analyte is unknown or the
    channel is required but missing/ambiguous.
    """
    rows = cats.db.doquery(
        "SELECT display_name, param_num, channel FROM hats.analyte_list "
        f"WHERE inst_num = {cats.inst_num} "
        f"AND display_name = '{analyte}';"
    ) or []
    if not rows:
        available = sorted(cats.analytes.keys())
        sys.exit(
            f"No analyte named {analyte!r} for CATS-{cats.site.upper()}. "
            f"Available: {available}"
        )

    by_channel = {(r['channel'] or None): int(r['param_num']) for r in rows}
    if channel is not None:
        channel = channel.lower().strip()
        if channel not in by_channel:
            sys.exit(
                f"Analyte {analyte!r} has no channel {channel!r} on "
                f"CATS-{cats.site.upper()}. Available channels: "
                f"{sorted(c for c in by_channel if c)}"
            )
        return by_channel[channel], channel

    if len(by_channel) > 1:
        sys.exit(
            f"Analyte {analyte!r} is present on multiple channels for "
            f"CATS-{cats.site.upper()}: {sorted(c for c in by_channel if c)}. "
            f"Pass --channel to disambiguate."
        )
    (only_channel, pnum), = by_channel.items()
    return pnum, only_channel


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_method_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--method', type=int, required=True, choices=sorted(METHOD_LABELS),
        help="detrend_method_num to apply (see list below).",
    )
    parser.add_argument(
        '--site', type=str, required=True,
        help=f"CATS site code. Valid: {sorted(CATS_Instrument.INST_NUM_BY_SITE)}.",
    )
    parser.add_argument(
        '--analyte', type=str, required=True,
        help="Analyte display_name from hats.analyte_list (e.g. N2O).",
    )
    parser.add_argument(
        '--channel', type=str, default=None,
        help="GC channel (q, f, cc, ...). Required if --analyte is ambiguous "
             "across channels.",
    )
    parser.add_argument(
        '--start', type=_parse_yyyymmdd, default=None,
        help="Start date, YYYYMMDD (default: Jan 1 of the current year).",
    )
    parser.add_argument(
        '--end', type=_parse_yyyymmdd, default=None,
        help="End date, YYYYMMDD (default: now).",
    )
    parser.add_argument(
        '-i', '--insert', action='store_true',
        help="Write the new smoothing method and recomputed mole fractions "
             "to the DB. Without this, only a dry-run preview is printed.",
    )
    parser.add_argument(
        '--recalc', action='store_true',
        help="After writing, also invoke 'cats_batch.py --fits -i' for this "
             "site/analyte/channel/date-range to recompute weekly cal fits "
             "and mole fractions from the new smoothing. Implies -i.",
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Print per-row detail while loading/smoothing.",
    )
    args = parser.parse_args()

    if args.recalc:
        args.insert = True

    start_date = args.start or f"{datetime.today().year}-01-01"
    end_date = args.end or datetime.today().strftime("%Y-%m-%d")

    cats = CATS_Instrument(site=args.site)
    pnum, channel = resolve_analyte(cats, args.analyte, args.channel)

    print(f"CATS-{cats.site.upper()}  analyte={args.analyte!r} (pnum={pnum}, "
          f"channel={channel!r})  {start_date} -> {end_date}")
    print(f"method {args.method}: {METHOD_LABELS[args.method]}")

    df = cats.load_data(
        pnum=pnum, channel=channel,
        start_date=start_date, end_date=end_date,
        verbose=args.verbose,
    )
    if df.empty:
        print("No data found for that analyte/date range.")
        return

    n_runs = df['run_time'].nunique()
    print(f"Loaded {len(df)} rows across {n_runs} run_time groups.")

    df['detrend_method_num'] = args.method
    df = cats.norm.merge_smoothed_data(df, detrend_method_num=args.method)
    df = cats.calc_mole_fraction(df)

    n_mf = df['mole_fraction'].notna().sum()
    print(f"Recomputed mole fractions: {n_mf}/{len(df)} rows non-null.")

    if not args.insert:
        print("Dry run (pass -i to write to the DB).")
        return

    cats.upsert_mole_fractions(df)
    print(f"Upserted {len(df)} rows into hats.ng_insitu_mole_fractions "
          f"(detrend_method_num={args.method}).")

    if args.recalc:
        cmd = [
            sys.executable, 'cats_batch.py',
            '-p', str(pnum), '-c', channel,
            '--site', cats.site,
            '-s', start_date, '-e', end_date,
            '--fits', '-i',
        ]
        if args.verbose:
            cmd.append('-v')
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
