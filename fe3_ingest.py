#!/usr/bin/env python3

# This script is intended to be run as a cron job.
# It performs a series of data ingestion and processing tasks.

import subprocess
from datetime import date

def get_previous_yymm():
    today = date.today()
    # subtract one month
    year = today.year
    month = today.month - 1
    if month == 0:
        month = 12
        year -= 1
    return f"{str(year)[2:]}{month:02d}"

def run_commands():
    cmds = [
        ["/hats/gc/itxbin/fe3_import.py"],
        ["/hats/gc/itxbin/fe3_export.py"],
        ["/hats/gc/itxbin/fe3_gcwerks2db.py"],    # default to past 30 days
        ["/hats/gc/itxbin/fe3_batch.py", "-p", "all", "-i", "-s", get_previous_yymm()],
    ]

    for cmd in cmds:
        # suppress output like > /dev/null 2>&1 for the first two
        if cmd[0].endswith("fe3_import.py") or cmd[0].endswith("fe3_export.py"):
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_commands()
