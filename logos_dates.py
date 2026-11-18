"""Shared start/end date options for the LOGOS loader, batch and ingest scripts.

Every <inst>_gcwerks2db.py, <inst>_batch.py and <inst>_ingest.py (M4, FE3, IE3, BLD1) takes the
same range options:

  -s/--start DATE   first time to process
  -e/--end DATE     last time to process, inclusive
  --all             the instrument's whole record
  --year YYYY       one calendar year (the same as -s YYYY-01-01 -e YYYY-12-31)

DATE is YYMM, YYYYMM, YYYY-MM, YYYYMMDD, YYYY-MM-DD, YYYY-MM-DD HH:MM[:SS], or the words 'start'
(instrument start) and 'end'/'now' (now). The end is inclusive at the precision given: 2605 is
through May 31 2026 23:59:59, 2026-05-15 is through the end of that day, a time is that moment.

With none of them the range is the default window: from the start of last month to now. All times
are UTC, as in the database.

    import logos_dates
    logos_dates.add_range_args(parser)
    args = parser.parse_args()
    start, end = logos_dates.resolve(args, inst_start=inst.start_date)   # pd.Timestamps, end never None
    logos_dates.sql(start)                                                # 'YYYY-MM-DD HH:MM:SS'
"""

import re
from datetime import datetime, timezone

import pandas as pd

SQL_FORMAT = "%Y-%m-%d %H:%M:%S"


def now_utc() -> pd.Timestamp:
    """Now in UTC, naive (the database's times are naive UTC), to the second."""
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).floor("s")


def default_start(now=None) -> pd.Timestamp:
    """The start of last month: the default window's first time."""
    now = now_utc() if now is None else pd.Timestamp(now)
    return (now.normalize().replace(day=1) - pd.offsets.MonthBegin(1)).normalize()


def _yy(yy: int) -> int:
    return 1900 + yy if yy >= 70 else 2000 + yy


def _parse(text, end: bool, inst_start=None):
    """One DATE as a Timestamp: the first moment of what it names, or with end=True the last."""
    if text is None:
        return None
    if isinstance(text, (pd.Timestamp, datetime)):
        return pd.Timestamp(text)
    s = str(text).strip()
    low = s.lower()
    if low == "start":
        if end:
            raise ValueError("'start' is not a valid end date")
        return None if inst_start is None else _parse(inst_start, False)
    if low in ("end", "now"):
        if not end:
            raise ValueError(f"{s!r} is not a valid start date")
        return now_utc()
    if re.fullmatch(r"\d{4}", s):                       # YYMM
        first, unit = pd.Timestamp(_yy(int(s[:2])), int(s[2:]), 1), "month"
    elif re.fullmatch(r"\d{6}", s):                     # YYYYMM
        first, unit = pd.Timestamp(int(s[:4]), int(s[4:]), 1), "month"
    elif re.fullmatch(r"\d{4}-\d{1,2}", s):             # YYYY-MM
        y, m = s.split("-")
        first, unit = pd.Timestamp(int(y), int(m), 1), "month"
    elif re.fullmatch(r"\d{8}", s):                     # YYYYMMDD
        first, unit = pd.Timestamp(int(s[:4]), int(s[4:6]), int(s[6:])), "day"
    elif re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", s):     # YYYY-MM-DD
        first, unit = pd.Timestamp(s), "day"
    else:
        try:
            return pd.Timestamp(s)                      # a date and time: that moment
        except ValueError:
            raise ValueError(f"unrecognised date {s!r} (use YYMM, YYYY-MM-DD or YYYY-MM-DD HH:MM)") from None
    if not end:
        return first
    if unit == "month":
        return first + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return first + pd.Timedelta(hours=23, minutes=59, seconds=59)


def parse_start(text, inst_start=None):
    """A start DATE as a Timestamp (None for None, or 'start' with no instrument start)."""
    return _parse(text, False, inst_start)


def parse_end(text):
    """An end DATE as the last Timestamp it includes (None for None)."""
    return _parse(text, True)


def resolve(args=None, start=None, end=None, all_data=False, year=None, inst_start=None, now=None):
    """(start, end) Timestamps for a run. Takes an argparse namespace from add_range_args() or the
    values themselves. --all gives (inst_start, now) (start None when the instrument start isn't
    known: everything); --year that year; otherwise -s/-e, each defaulting to the default window's
    (start of last month, now). end is never None."""
    if args is not None:
        start = getattr(args, "start", None)
        end = getattr(args, "end", None)
        all_data = bool(getattr(args, "all", False))
        year = getattr(args, "year", None)
    if sum(x is not None and x is not False for x in (all_data or None, year, start or end)) > 1:
        raise ValueError("use one of --all, --year or -s/-e")
    now = now_utc() if now is None else pd.Timestamp(now)
    if all_data:
        return parse_start(inst_start) if inst_start is not None else None, now
    if year is not None:
        y = int(year)
        return pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31, 23, 59, 59)
    s = parse_start(start, inst_start) if start is not None else default_start(now)
    e = parse_end(end) if end is not None else now
    if s is not None and e < s:
        raise ValueError(f"end {e} is before start {s}")
    return s, e


def sql(ts):
    """A Timestamp as 'YYYY-MM-DD HH:MM:SS' (None stays None)."""
    return None if ts is None else pd.Timestamp(ts).strftime(SQL_FORMAT)


def yymm(ts):
    """A Timestamp's month as GCwerks' YYMM."""
    return pd.Timestamp(ts).strftime("%y%m")


DATE_HELP = "YYMM, YYYY-MM-DD or 'YYYY-MM-DD HH:MM'"


def add_range_args(parser, start_aliases=(), end_aliases=(), year=True):
    """Add -s/--start, -e/--end, --all and --year to an argparse parser. *_aliases are older
    option strings kept working (e.g. '--start-date')."""
    parser.add_argument("-s", "--start", *start_aliases, dest="start", metavar="DATE",
                        help=f"first time to process: {DATE_HELP}, or 'start' for the instrument start "
                             "(default: the start of last month)")
    parser.add_argument("-e", "--end", *end_aliases, dest="end", metavar="DATE",
                        help=f"last time to process, inclusive: {DATE_HELP} (2605 = through May 31), "
                             "or 'end' (default: now)")
    parser.add_argument("--all", action="store_true", dest="all", help="the instrument's whole record")
    if year:
        parser.add_argument("--year", type=int, dest="year", metavar="YYYY", help="one calendar year")
    return parser


def range_cli_args(start, end):
    """-s/-e arguments for a sub-script, from resolved Timestamps (start None = 'start')."""
    return ["-s", sql(start) if start is not None else "start", "-e", sql(end)]
