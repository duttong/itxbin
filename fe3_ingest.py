#!/usr/bin/env python3
"""Import, export, load and recalculate FE3 data (run daily from cron).

    fe3_ingest.py                    # the default window: from the start of last month to now
    fe3_ingest.py -s 2603 -e 2605    # March through May 2026
    fe3_ingest.py --year 2025 | --all

The range (logos_dates options) goes to fe3_gcwerks2db.py and fe3_batch.py, so both cover the same
runs. fe3_import.py and fe3_export.py always do their whole job.
"""

import argparse
import subprocess

import logos_dates


def run_commands(range_args):
    cmds = [
        ["/hats/gc/itxbin/fe3_import.py"],
        ["/hats/gc/itxbin/fe3_export.py", "--flagged"],
        ["/hats/gc/itxbin/fe3_gcwerks2db.py", "--flagged", *range_args],
        ["/hats/gc/itxbin/fe3_batch.py", "-p", "all", "-i", *range_args],
    ]

    for cmd in cmds:
        # suppress output like > /dev/null 2>&1 for the first two
        if cmd[0].endswith("fe3_import.py") or cmd[0].endswith("fe3_export.py"):
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Ingest and process FE3 GC data.")
    logos_dates.add_range_args(parser)
    args = parser.parse_args()
    try:
        start, end = logos_dates.resolve(args)
    except ValueError as e:
        parser.error(str(e))
    run_commands(["--all"] if args.all else logos_dates.range_cli_args(start, end))


if __name__ == "__main__":
    main()
