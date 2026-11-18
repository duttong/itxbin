"""Shared pieces of m4_ingest.py, fe3_ingest.py and ie3_ingest.py: the --steps option and the YYMM
gcupdate starts from.

Each ingest script lists its steps in order and which run by default (the daily cron run). --steps
picks a subset by name, comma separated; they still run in the script's order. The LOGOS Load page's
jobs (logos_jobs.py) use it, e.g. a reload is `--steps load,batch -s .. -e ..`.
"""

import datetime
from pathlib import Path

import logos_dates


def add_steps_arg(parser, steps, default):
    """--steps for an argparse parser. steps: every step name in run order; default: the daily set."""
    parser.add_argument("--steps", default=",".join(default), metavar="LIST",
                        help=f"steps to run, comma separated, from: {', '.join(steps)} "
                             f"(default: {','.join(default)})")


def parse_steps(text, steps):
    """The set of step names in a --steps value; ValueError for an unknown one."""
    wanted = {s.strip().lower() for s in str(text).split(",") if s.strip()}
    bad = wanted - set(steps)
    if bad:
        raise ValueError(f"unknown step(s) {', '.join(sorted(bad))}; choose from {', '.join(steps)}")
    if not wanted:
        raise ValueError("no steps given")
    return wanted


def last_month_yymm(today=None):
    today = today or datetime.date.today()
    return (today.replace(day=1) - datetime.timedelta(days=1)).strftime("%y%m")


def gcupdate_yymm(gc_dir, start, all_data):
    """The YYMM gcupdate starts from: last month, or the range's month when that is earlier, or the
    first year folder of gc_dir for --all. gcupdate YYMM continues to the present."""
    yymm = last_month_yymm()
    if all_data:
        years = sorted(int(p.name) for p in Path(gc_dir).glob("[0-9][0-9]") if p.is_dir())
        years = sorted(years, key=lambda y: (y < 70, y))     # 98, 99, 00, 01, ...
        return f"{years[0]:02d}01" if years else yymm
    if start is None:
        return yymm
    return min(yymm, logos_dates.yymm(start), key=lambda m: (int(m[:2]) < 70, m))


def range_args(args, start, end):
    """The -s/-e (or --all) arguments that hand an ingest's range to its loader and batch."""
    return ["--all"] if args.all else logos_dates.range_cli_args(start, end)
