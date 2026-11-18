#!/usr/bin/env python3
"""
m4_ingest.py

Copy recent bdMMDDYY directories from RAW and bdMMDDYY.txt/.xl files from GSPC into INCOMING,
remove older items by name, then run gcwerks and itxbin processing steps.
Parallelize RAW directory sync for speed-up if IO allows.
"""
import argparse
import datetime
import logging
import re
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import ingest_common
import logos_dates
import logos_lock


def parse_bd_date(name: str) -> datetime.date | None:
    """
    Parse names like bdMMDDYY, returning a date or None if it doesn't match or is invalid.
    """
    m = re.match(r"^bd(\d{2})(\d{2})(\d{2})", name)
    if not m:
        return None
    mm, dd, yy = m.groups()
    year = 2000 + int(yy)
    try:
        return datetime.date(year, int(mm), int(dd))
    except ValueError:
        return None


def clean_incoming(incoming: Path, threshold: datetime.date):
    """
    Remove any file or directory in `incoming` whose bd date is older than threshold.
    """
    for entry in incoming.iterdir():
        d = parse_bd_date(entry.name)
        if d and d < threshold:
            if entry.is_dir():
                shutil.rmtree(entry)
                logging.info(f"Removed old directory: {entry}")
            else:
                entry.unlink()
                logging.info(f"Removed old file: {entry}")


def _copy_directory(entry: Path, incoming: Path):
    """
    Helper to copy one raw subdirectory tree into incoming.
    """
    dest = incoming / entry.name
    logging.info(f"Syncing directory {entry} → {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(entry, dest)


def sync_raw(raw_dir: Path, incoming: Path, threshold: datetime.date, max_workers: int = 4):
    """
    Copy bdMMDDYY directories from RAW → INCOMING in parallel if date >= threshold.
    Overwrite existing.

    Args:
        raw_dir: source Path
        incoming: destination Path
        threshold: earliest date to include
        max_workers: number of threads for parallel copy
    """
    to_sync = []
    for entry in raw_dir.iterdir():
        if entry.is_dir():
            d = parse_bd_date(entry.name)
            if d and d >= threshold:
                # A run dir with no .D data (an aborted sequence, e.g. bd110526)
                # makes GCwerks' chemstation import stop at it, so every later
                # run dir is silently never imported. Skip it, and drop any
                # copy already in incoming.
                if not any(entry.glob("*.D")):
                    dest = incoming / entry.name
                    if dest.is_dir():
                        shutil.rmtree(dest)
                        logging.info(f"Removed empty run directory: {dest}")
                    logging.info(f"Skipping {entry}: no .D data")
                    continue
                to_sync.append(entry)
    if not to_sync:
        return
    # Parallel copy; adjust max_workers based on available tasks
    workers = min(max_workers, len(to_sync))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for entry in to_sync:
            executor.submit(_copy_directory, entry, incoming)


def sync_gspc(gspc_dir: Path, incoming: Path, threshold: datetime.date):
    """
    Copy bdMMDDYY.txt or .xl files from GSPC → INCOMING if date >= threshold.
    Overwrite existing.
    """
    for entry in gspc_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in ('.txt', '.xl'):
            d = parse_bd_date(entry.stem)
            if d and d >= threshold:
                dest = incoming / entry.name
                logging.info(f"Copying file {entry} → {dest}")
                shutil.copy2(entry, dest)


STEPS = ("sync", "import", "update", "load", "batch")


def run_commands(gcd: Path, yymm: str, range_args: list[str], steps: set):
    """
    Run the series of gcwerks and itxbin commands against the INCOMING directory.

    yymm is GCwerks' YYMM form of last month (e.g. "2606"), passed as a bare
    positional arg to gcupdate -- the same convention already used in
    gcwerks_import.py for FE3/IE3/BLD1. gcupdate's own incremental default
    doesn't reliably re-scan stragglers from the tail end of the previous
    month (observed: an overnight run starting the evening of the last day
    of the month was silently skipped), so it's passed explicitly here.

    range_args (-s/-e or --all) go to the loader and the batch, so both cover the same range.
    steps: which of import, update, load, batch to run (sync is done by main).
    """
    cmds = [
        ("import", ["/hats/gc/gcwerks-3/bin/gcimport",    "-gcdir", str(gcd)]),
        ("import", ["/hats/gc/gcwerks-3/bin/run-index",   "-gcdir", str(gcd)]),
        ("import", ["/hats/gc/itxbin/m4_samplogs.py",     "-i"]),
        ("update", ["/hats/gc/gcwerks-3/bin/gcupdate",    "-gcdir", str(gcd), yymm]),
        ("update", ["/hats/gc/gcwerks-3/bin/gccalc",      "-gcdir", str(gcd)]),
        ("load",   ["/hats/gc/itxbin/m4_gcwerks2db.py",   "-x", "--flagged", *range_args]),
        ("batch",  ["/hats/gc/itxbin/m4_batch.py",        "-p", "all", "-i", *range_args]),
    ]
    for step, cmd in cmds:
        if step not in steps:
            continue
        logging.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Ingest and process m4 GC data.")
    parser.add_argument("--days", "-d", type=int, default=14,
                        help="Retention period in days (default: 14)")
    parser.add_argument("--incoming", type=Path,
                        default=Path("/hats/gc/m4/chemstation"),
                        help="Destination dir for ingest (default: /hats/gc/m4/chemstation)")
    parser.add_argument("--raw", type=Path,
                        default=Path("/hats/gc/m4/MassHunter/GCMS/1/data"),
                        help="Source RAW dir (default: /hats/gc/m4/MassHunter/GCMS/1/data)")
    parser.add_argument("--gspc", type=Path,
                        default=Path("/hats/gc/m4/MassHunter/GCMS/M4 GSPC Files"),
                        help="Source GSPC dir (default: /hats/gc/m4/MassHunter/GCMS/M4 GSPC Files)")
    logos_dates.add_range_args(parser)
    ingest_common.add_steps_arg(parser, STEPS, STEPS)
    args = parser.parse_args()
    try:
        start, end = logos_dates.resolve(args)
        steps = ingest_common.parse_steps(args.steps, STEPS)
    except ValueError as e:
        parser.error(str(e))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    logging.info(f"M4 ingest: steps {','.join(s for s in STEPS if s in steps)}; "
                 f"range {'all' if args.all else f'{start} to {end}'}")
    lock = logos_lock.acquire("m4", log=logging.info)      # held until exit

    today = datetime.date.today()
    threshold = today - datetime.timedelta(days=args.days)

    # GCwerks YYMM for gcupdate: last calendar month (e.g. "2606"), so it always
    # re-covers all of last month plus this month to date, regardless of --days;
    # earlier when the range starts earlier (--all: the first year folder).
    yymm = ingest_common.gcupdate_yymm('/hats/gc/m4', start, args.all)
    range_args = ingest_common.range_args(args, start, end)

    if "sync" in steps:
        # ensure incoming exists
        args.incoming.mkdir(parents=True, exist_ok=True)
        sync_raw(args.raw, args.incoming, threshold)
        sync_gspc(args.gspc, args.incoming, threshold)
    run_commands('/hats/gc/m4', yymm, range_args, steps)
    if "sync" in steps:
        # remove older directories/files
        clean_incoming(args.incoming, threshold)


if __name__ == "__main__":
    main()
