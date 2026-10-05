#!/usr/bin/env python3
"""
m4_thermocouples.py [bdfile] [-o out.png]

Plot the M4 GSPC trap thermocouples (temps_*.csv, therm0/therm1) against the
cryogen / valve timing in a GSPC run log (bd*.txt).

Top panel     whole run, cryogen-on and GC-cryogen-on spans shaded.
Bottom panel  every injection overlaid (thin lines) with the mean (thick),
              aligned on "Sample valve open", with the trap_cold and trap_hot
              windows shaded:
                trap_cold  sample valve open -> 30 s before sample valve closed
                trap_hot   8 to 13 min after sample valve open
              (same definitions as hats.ng_ancillary_data.trap_cold/trap_hot,
              loaded by m4_samplogs.py)

<bdfile> is a path or a bare name such as bd100126 / bd100126.txt (looked up in
the GSPC directory). If omitted, use the latest bdMMDDYY.txt file by its
filename date in the GSPC directory. Output defaults to <bdname>_thermocouples.png.
"""
import argparse
from datetime import datetime
import re
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

GSPC_DIR = Path("/hats/gc/m4/MassHunter/GCMS/M4 GSPC Files")
EVENT_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+: (.*)")


def read_log(path):
    ev = []
    with open(path, errors="replace") as log:
        for line in log:
            m = EVENT_RE.match(line)
            if m:
                ev.append((pd.Timestamp(m[1]), m[2].strip()))
    return ev


def spans(ev, on, off):
    out, start = [], None
    for ts, e in ev:
        if e == on and start is None:
            start = ts
        elif e == off and start is not None:
            out.append((start, ts))
            start = None
    return out


def read_temps(directory, t0, t1):
    """All temps_*.csv rows overlapping [t0, t1], de-duplicated."""
    dfs = []
    for path in sorted(directory.glob("temps_*.csv")):
        frame = pd.read_csv(path)
        # Edited copies can contain times alone. Never invent a date for them.
        frame["datetime"] = pd.to_datetime(
            frame["datetime"], format="ISO8601", errors="coerce"
        )
        dfs.append(frame.dropna(subset=["datetime"]))
    if not dfs:
        raise SystemExit(f"No temps_*.csv files found in {directory}")
    t = pd.concat(dfs).drop_duplicates("datetime").sort_values("datetime")
    return t[(t.datetime >= t0 - pd.Timedelta(minutes=5)) & (t.datetime <= t1 + pd.Timedelta(minutes=5))].reset_index(drop=True)


def trap_values(ev, t, opens):
    """Per-injection trap_cold / trap_hot (therm1 means), as loaded into ng_ancillary_data."""
    minute = pd.Timedelta(minutes=1)
    cold, hot = [], []
    for o in opens:
        closed = [ts for ts, e in ev if e == "Sample valve closed" and ts > o]
        if not closed:
            continue
        ce = closed[0] - pd.Timedelta(seconds=30)
        for out, (a, b, expected) in ((cold, (o, ce, (ce - o).total_seconds() / 10)),
                                      (hot, (o + 8 * minute, o + 13 * minute, 30))):
            w = t.loc[(t.datetime >= a) & (t.datetime <= b), "therm1"]
            if len(w) >= 0.8 * expected:
                out.append(w.mean())
    return np.array(cold), np.array(hot)


def fmt_stats(v):
    """mean ± sd (min, max), 2 decimals."""
    if len(v) == 0:
        return "(no data)"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.2f} ± {sd:.2f} ({v.min():.2f}, {v.max():.2f}) °C"


def build_figure(bd):
    """Build the plot without selecting a backend or writing a file."""
    ev = read_log(bd)
    if not ev:
        raise SystemExit(f"No timestamped events in {bd}")
    t = read_temps(bd.parent, ev[0][0], ev[-1][0])
    if t.empty:
        raise SystemExit(f"No temps_*.csv data overlaps {bd.name} ({ev[0][0]} - {ev[-1][0]})")

    cryo = spans(ev, "Activated cryogen", "Deactivated cryogen")
    gcc = spans(ev, "Activated GC cryogen", "Deactivated GC cryogen")
    opens = [ts for ts, e in ev if e == "Sample valve open"]
    ref = t.datetime.iloc[0].normalize()
    tt = (t.datetime - ref).dt.total_seconds().values
    grid = np.arange(-240, 1080, 5.0)  # seconds relative to sample valve open

    fig = Figure(figsize=(14, 10))
    ax = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]})

    a = ax[0]
    a.plot(t.datetime, t.therm0, lw=.9, color="tab:red", label="therm0")
    a.plot(t.datetime, t.therm1, lw=.9, color="tab:blue", label="therm1")
    for i, (s, e) in enumerate(cryo):
        a.axvspan(s, e, color="tab:cyan", alpha=.25, lw=0, label="cryogen on" if i == 0 else None)
    for i, (s, e) in enumerate(gcc):
        a.axvspan(s, e, color="tab:orange", alpha=.25, lw=0, label="GC cryogen on" if i == 0 else None)
    a.set_xlim(t.datetime.min(), t.datetime.max())
    a.set_ylabel("Temperature (°C)")
    a.grid(alpha=.3)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    a.set_title(f"M4 GSPC temperatures and cryogen timing, run {bd.stem}")
    a.legend(loc="upper right", ncol=4)

    a = ax[1]
    stack = {0: [], 1: []}
    for o in opens:
        o0 = (o - ref).total_seconds()
        if o0 + grid[0] < tt[0] or o0 + grid[-1] > tt[-1]:
            continue
        for k, c in ((0, "therm0"), (1, "therm1")):
            y = np.interp(o0 + grid, tt, t[c].values)
            stack[k].append(y)
            a.plot(grid / 60, y, lw=.4, alpha=.35, color=("tab:red", "tab:blue")[k])
    n = len(stack[0])
    if n == 0:
        raise SystemExit("No complete injections with temperature coverage")
    for k, c, name in ((0, "tab:red", "therm0"), (1, "tab:blue", "therm1")):
        a.plot(grid / 60, np.array(stack[k]).mean(0), lw=3, color=c, label=f"{name} mean (n={n})")

    for name, e in (("Sample valve open", "Sample valve open"), ("Precolumn IN", "Precolumn IN line"),
                    ("Sample valve closed", "Sample valve closed"), ("GC cryogen off", "Deactivated GC cryogen")):
        d = []
        for o in opens:
            c = [(ts - o).total_seconds() for ts, e2 in ev if e2 == e and -1 <= (ts - o).total_seconds() < 1000]
            if c:
                d.append(min(c))
        if d:
            x = np.median(d) / 60
            a.axvline(x, color="k", lw=.7, ls=":")
            a.text(x, 212, name, rotation=90, va="top", ha="right", fontsize=8)
    closes = [min((ts - o).total_seconds() for ts, e in ev if e == "Sample valve closed" and ts > o)
              for o in opens if any(e == "Sample valve closed" and ts > o for ts, e in ev)]
    cold_end = (np.median(closes) - 30) / 60 if closes else 4.0
    cold, hot = trap_values(ev, t, opens)
    a.axvspan(0, cold_end, color="tab:cyan", alpha=.12, lw=0, label=f"trap_cold {fmt_stats(cold)}")
    a.axvspan(8, 13, color="tab:orange", alpha=.15, lw=0, label=f"trap_hot {fmt_stats(hot)}")
    a.set_xlabel("Minutes since sample valve open")
    a.set_ylabel("Temperature (°C)")
    a.grid(alpha=.3)
    a.set_title(f"All injections overlaid, aligned on sample valve open (n={n})")
    a.legend(loc="lower right")
    a.set_xlim(grid[0] / 60, grid[-1] / 60)
    a.set_ylim(-190, 215)

    fig.tight_layout()
    return fig, n


def make_figure(bd, out):
    fig, n = build_figure(bd)
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}  ({n} injections)")


def bd_file_for_run(directory, run_time):
    """Locate a selected run's log in the active directory or year archive."""
    date = pd.Timestamp(str(run_time).split(" (")[0])
    name = f"bd{date.strftime('%m%d%y')}.txt"
    for parent in (directory, directory / str(date.year)):
        path = parent / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {name} in {directory} or its {date.year} archive")


def latest_bd_file(directory):
    """Return the bdMMDDYY.txt file with the newest valid filename date."""
    candidates = []
    for path in directory.glob("bd*.txt"):
        match = re.fullmatch(r"bd(\d{6})\.txt", path.name)
        if not match or not path.is_file():
            continue
        try:
            date = datetime.strptime(match[1], "%m%d%y").date()
        except ValueError:
            continue
        candidates.append((date, path))
    if not candidates:
        raise SystemExit(f"No valid bdMMDDYY.txt files found in {directory}")
    return max(candidates, key=lambda item: item[0])[1]


def main():
    ap = argparse.ArgumentParser(description="Plot M4 GSPC thermocouple temperatures against a run log (bd*.txt).")
    ap.add_argument("bdfile", nargs="?", help="bd log file path or name (default: newest bdMMDDYY.txt in the GSPC directory)")
    ap.add_argument("-o", "--output", help="output png (default <bdname>_thermocouples.png)")
    args = ap.parse_args()

    if args.bdfile is None:
        bd = latest_bd_file(GSPC_DIR)
    else:
        bd = Path(args.bdfile)
        if bd.suffix != ".txt":
            bd = bd.with_name(bd.name + ".txt")
        if not bd.exists():
            bd = GSPC_DIR / bd.name
        if not bd.exists():
            raise SystemExit(f"Cannot find {args.bdfile}")
    make_figure(bd, args.output or f"{bd.stem}_thermocouples.png")


if __name__ == "__main__":
    main()
