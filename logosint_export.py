#!/usr/bin/env python3
"""Write logosint integrations of a CATS site as gcexport-style per-gas CSVs.

Runs `logosint integrate` over a month range and writes one CSV per gas to
/hats/gc/cats_results/logosint/ (brw_F11.csv, brw_N2O.csv, ... as cats_export.py
names them), with gcexport's columns: time, port, <gas>_rt, <gas>_ht, <gas>_area.
cats_gcwerks2db.py --source logosint loads them.

- A peak logosint found: its retention time, height and area.
- No peak in the window (status missing): 0.00, 0.00, 0.0, which the loader treats
  as "not measured" (as it does GCwerks' zeros).
- A run logosint refused (its parameter file uses a command logosint doesn't have):
  no row for that gas at all, so the loader leaves the DB's value alone.
- No flag marks: GCwerks' flags (tag 324) are not carried into logosint results.

By default the months integrated replace the same months in the existing files and
everything else in them is kept; --all rebuilds the files from --start (default the
first year in the GCwerks folder). A sidecar <site>_logosint.json records the logosint
version, parameter set and ranges written.

  logosint_export.py brw                         # last two months to now
  logosint_export.py brw --start 2106 --end 2612
  logosint_export.py brw --all --start 2106      # files hold 2021-06 on only
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

OUT_DIR = Path("/hats/gc/cats_results/logosint")
LOGOSINT = "/hats/gc/logosint/logosint"

# As cats_export.py (CATS_Export.repmols): GCwerks peak name -> file name part.
REPMOLS = {
    "n2o_q": "N2O", "sf6_q": "SF6", "n2o_a": "N2Oa", "cfc12_a": "F12", "h1211_a": "H1211a",
    "cfc11_a": "F11a", "cfc113_a": "F113a", "cfc12_f": "F12f", "h1211_f": "H1211", "cfc11_f": "F11",
    "cfc113_f": "F113", "chcl3_f": "CHCl3", "ch3ccl3_f": "MC", "ccl4_f": "CCl4", "ocs_cc": "OCS",
    "hcfc22_cc": "HCFC22", "ch3cl_cc": "CH3Cl", "cfc12_cc": "F12cc", "hcfc142b_cc": "HCFC142b",
    "ch3br_cc": "CH3Br", "h1211_cc": "H1211cc",
}


def yymm(d):
    return d.strftime("%y%m")


def month_start(s):
    """yymm -> Timestamp of the month's start."""
    yy, mm = int(s[:2]), int(s[2:])
    return pd.Timestamp(1900 + yy if yy >= 70 else 2000 + yy, mm, 1)


def integrate(site, start, end, channels, pset, binary):
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tf:
        out = tf.name
    cmd = [binary, "integrate", "--site", site, "--start", start, "--end", end,
           "--channel", ",".join(str(c) for c in channels), "--out", out]
    cmd += ["--files"] if pset == "files" else ["--set", pset]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        sys.exit(f"logosint integrate failed ({r.returncode})")
    head = [l.rstrip() for l in open(out) if l.startswith("#")]
    df = pd.read_csv(out, sep="\t", comment="#", dtype={"run": str, "port": str})
    os.unlink(out)
    return df, head


def to_frames(df):
    """gas -> DataFrame in gcexport layout (refused runs left out)."""
    df = df[df["status"].isin(["ok", "missing"])].copy()
    parts = df["run"].str.split(".")
    df["time"] = pd.to_datetime(parts.str[0] + parts.str[1], format="%y%m%d%H%M")
    df["port"] = pd.to_numeric(df["port"], errors="coerce")
    df = df[df["port"].notna()]
    out = {}
    for gas, g in df.groupby("gas"):
        ok = g["status"].eq("ok")
        f = pd.DataFrame({
            "time": g["time"], "port": g["port"].astype(int),
            f"{gas}_rt": g["rt"].where(ok, 0.0), f"{gas}_ht": g["height"].where(ok, 0.0),
            f"{gas}_area": g["area"].where(ok, 0.0),
        })
        out[gas] = f.sort_values(["time", "port"])
    return out


def write_csv(path, frame, gas):
    """gcexport's text layout: 'time, port, X_rt, X_ht, X_area', values '2026-10-01 00:18, 4, 413.00, 37336.14, 889337.3'."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(f"time, port, {gas}_rt, {gas}_ht, {gas}_area\n")
        for t, p, rt, ht, area in frame[["time", "port", f"{gas}_rt", f"{gas}_ht", f"{gas}_area"]].itertuples(index=False):
            f.write(f"{t:%Y-%m-%d %H:%M}, {int(p)}, {rt:.2f}, {ht:.2f}, {area:.1f}\n")
    os.chmod(tmp, 0o664)
    os.replace(tmp, path)


def read_existing(path, gas):
    if not path.exists():
        return None
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d["time"] = pd.to_datetime(d["time"])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", help="CATS GCwerks folder: brw, ...")
    ap.add_argument("--start", help="first month, yymm (default: two months ago; with --all, the folder's first year)")
    ap.add_argument("--end", help="month after the last, yymm (default: next month)")
    ap.add_argument("--all", action="store_true", help="rebuild the files from --start instead of replacing months in them")
    ap.add_argument("--channels", default="0,1,2", help="channels to integrate (default 0,1,2)")
    ap.add_argument("--set", default="logosint", help="parameter set: logosint (default), gcwerks, or files")
    ap.add_argument("--out", default=str(OUT_DIR), help=f"output folder (default {OUT_DIR})")
    ap.add_argument("--logosint", default=LOGOSINT, help=f"logosint binary (default {LOGOSINT})")
    a = ap.parse_args()

    today = date.today()
    end = a.end or yymm((pd.Timestamp(today) + pd.offsets.MonthBegin(1)).date())
    if a.start:
        start = a.start
    elif a.all:
        years = sorted(int(p.name) for p in Path(f"/hats/gc/{a.site}").glob("[0-9][0-9]") if p.is_dir())
        first = min((y + 2000 if y < 70 else y + 1900) for y in years)
        start = f"{first % 100:02d}01"
    else:
        start = yymm((pd.Timestamp(today) - pd.DateOffset(months=2)).date())
    t0, t1 = month_start(start), month_start(end)
    channels = [int(c) for c in a.channels.split(",")]
    out_dir = Path(a.out)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        # OneFS gives new folders here an ACL that blocks creating files despite the mode bits;
        # a chmod replaces it with the plain mode.
        os.chmod(out_dir, 0o775)

    df, head = integrate(a.site, start, end, channels, a.set, a.logosint)
    frames = to_frames(df)
    written = []
    for gas, new in frames.items():
        path = out_dir / f"{a.site}_{REPMOLS.get(gas.lower(), gas.upper())}.csv"
        old = None if a.all else read_existing(path, gas)
        if old is not None:
            keep = old[(old["time"] < t0) | (old["time"] >= t1)]
            new = pd.concat([keep, new], ignore_index=True).sort_values(["time", "port"])
        new = new.drop_duplicates(["time", "port"], keep="last")
        write_csv(path, new, gas)
        written.append((path.name, len(new)))

    side = out_dir / f"{a.site}_logosint.json"
    meta = {}
    if side.exists() and not a.all:
        try:
            meta = json.loads(side.read_text())
        except ValueError:
            meta = {}
    meta.setdefault("writes", []).append({
        "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "range": [start, end], "rebuild": a.all, "channels": channels, "set": a.set,
        "logosint": head[0] if head else "", "sources": head[1:], "by": os.environ.get("USER", ""),
    })
    meta["writes"] = meta["writes"][-200:]
    tmp = side.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=1) + "\n")
    os.replace(tmp, side)
    try:
        os.chmod(side, 0o664)
    except PermissionError:
        pass

    refused = int((df["status"] == "refused").sum())
    print(f"{a.site} {start}..{end}: " + ", ".join(f"{n} ({k} rows)" for n, k in written) + f" -> {out_dir}")
    if refused:
        print(f"{refused} gas rows refused (unsupported commands) and left out; the loader keeps the DB's values for them")


if __name__ == "__main__":
    main()
