#!/usr/bin/env python3
"""Import, export, load and recalculate FE3 data (run daily from cron).

    fe3_ingest.py                    # the default window: from the start of last month to now
    fe3_ingest.py -s 2603 -e 2605    # March through May 2026
    fe3_ingest.py --year 2025 | --all

The range (logos_dates options) goes to fe3_gcwerks2db.py and fe3_batch.py, so both cover the same
runs. fe3_import.py and fe3_export.py always do their whole job. --steps picks steps (update =
gcupdate from the range's month + gccalc, off by default); the run holds logos_lock 'fe3'.
"""

import argparse
import subprocess

import ingest_common
import logos_dates
import logos_lock


STEPS = ("import", "update", "export", "load", "batch")
DAILY = ("import", "export", "load", "batch")     # fe3_import.py runs gcupdate/gccalc itself when it imports
GC_DIR = "/hats/gc/fe3"
GCWERKS = "/hats/gc/gcwerks-3/bin"


def run_commands(range_args, steps, yymm):
    cmds = [
        ("import", ["/hats/gc/itxbin/fe3_import.py"]),
        ("update", [f"{GCWERKS}/gcupdate", "-gcdir", GC_DIR, yymm]),
        ("update", [f"{GCWERKS}/gccalc", "-gcdir", GC_DIR]),
        ("export", ["/hats/gc/itxbin/fe3_export.py", "--flagged"]),
        ("load",   ["/hats/gc/itxbin/fe3_gcwerks2db.py", "--flagged", *range_args]),
        ("batch",  ["/hats/gc/itxbin/fe3_batch.py", "-p", "all", "-i", *range_args]),
    ]

    for step, cmd in cmds:
        if step not in steps:
            continue
        print("Running: " + " ".join(cmd), flush=True)
        # suppress output like > /dev/null 2>&1 for the first two
        if cmd[0].endswith("fe3_import.py") or cmd[0].endswith("fe3_export.py"):
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Ingest and process FE3 GC data.")
    logos_dates.add_range_args(parser)
    ingest_common.add_steps_arg(parser, STEPS, DAILY)
    args = parser.parse_args()
    try:
        start, end = logos_dates.resolve(args)
        steps = ingest_common.parse_steps(args.steps, STEPS)
    except ValueError as e:
        parser.error(str(e))
    print(f"FE3 ingest: steps {','.join(s for s in STEPS if s in steps)}; "
          f"range {'all' if args.all else f'{start} to {end}'}", flush=True)
    lock = logos_lock.acquire("fe3")      # held until exit
    run_commands(ingest_common.range_args(args, start, end), steps,
                 ingest_common.gcupdate_yymm(GC_DIR, start, args.all))


if __name__ == "__main__":
    main()
