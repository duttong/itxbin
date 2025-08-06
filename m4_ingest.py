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


def run_commands(gcd: Path):
    """
    Run the series of gcwerks and itxbin commands against the INCOMING directory.
    """
    cmds = [
        ["/hats/gc/gcwerks-3/bin/gcimport",    "-gcdir", str(gcd)],
        ["/hats/gc/gcwerks-3/bin/run-index",   "-gcdir", str(gcd)],
        ["/hats/gc/itxbin/m4_samplogs.py",     "-i"],
        ["/hats/gc/gcwerks-3/bin/gcupdate",    "-gcdir", str(gcd)],
        ["/hats/gc/gcwerks-3/bin/gccalc",      "-gcdir", str(gcd)],
        ["/hats/gc/itxbin/m4_gcwerks2db.py",   "-x"],
        ["/hats/gc/itxbin/m4_batch.py",       "-p", "all", "-i"],
    ]
    for cmd in cmds:
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    today = datetime.date.today()
    threshold = today - datetime.timedelta(days=args.days)

    # ensure incoming exists
    args.incoming.mkdir(parents=True, exist_ok=True)

    clean_incoming(args.incoming, threshold)
    sync_raw(args.raw, args.incoming, threshold)
    sync_gspc(args.gspc, args.incoming, threshold)
    run_commands(args.incoming)


if __name__ == "__main__":
    main()
