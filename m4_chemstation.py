#!/usr/bin/env python3
"""
m4_chemstation.py

Reimport a single M4 chemstation directory (e.g. bd062626) that has already
been imported once. Removes the stale chromatograms, .run-index entries, and
sample.log rows for that directory's date, then re-runs the standard M4
ingest/processing pipeline (gcimport, run-index, m4_samplogs.py, gcupdate,
gccalc, m4_gcwerks2db.py, m4_batch.py) so it re-integrates from scratch.

The date is inferred from the directory name (bdMMDDYY), not from today's
date, so this works for old chemstation directories, not just recent ones.

Usage:
    m4_chemstation.py bd062626
    m4_chemstation.py /hats/gc/m4/chemstation/bd062626
"""
import argparse
import datetime
import logging
import subprocess
from pathlib import Path

GC_DIR = Path('/hats/gc/m4')
CHEMSTATION_DIR = GC_DIR / 'chemstation'


def parse_bd_date(name: str) -> datetime.date | None:
    """Parse names like bdMMDDYY, returning a date or None if invalid."""
    import re
    m = re.match(r"^bd(\d{2})(\d{2})(\d{2})", name)
    if not m:
        return None
    mm, dd, yy = m.groups()
    year = 2000 + int(yy)
    try:
        return datetime.date(year, int(mm), int(dd))
    except ValueError:
        return None


def find_chromatograms(yy: str, yymmdd: str) -> list[Path]:
    chrom_dir = GC_DIR / yy / 'chromatograms' / 'channel0'
    if not chrom_dir.is_dir():
        return []
    return sorted(p for p in chrom_dir.iterdir() if p.name.startswith(f"{yymmdd}."))


def find_runindex_lines(yymmdd: str) -> tuple[list[str], list[str]]:
    """Return (kept_lines, removed_lines) from .run-index, preserving order."""
    runindex = GC_DIR / '.run-index'
    lines = runindex.read_text().splitlines(keepends=True)
    kept, removed = [], []
    for line in lines:
        if line.strip().startswith(f"{yymmdd}."):
            removed.append(line)
        else:
            kept.append(line)
    return kept, removed


def find_samplelog_rows(date_obj: datetime.date, yymmdd: str) -> Path | None:
    """Return the sample.log file for date_obj's year, if it exists."""
    file_name = f"{str(date_obj.year)[2:]}01"
    log_path = GC_DIR / 'logs' / 'sample.log' / file_name
    return log_path if log_path.is_file() else None


def count_samplelog_rows(log_path: Path, yymmdd: str) -> int:
    count = 0
    with open(log_path) as f:
        next(f, None)  # header
        for line in f:
            if line.split('\t', 1)[0] == yymmdd:
                count += 1
    return count


def remove_samplelog_rows(log_path: Path, yymmdd: str):
    lines = log_path.read_text().splitlines(keepends=True)
    header, rest = lines[0], lines[1:]
    kept = [line for line in rest if line.split('\t', 1)[0] != yymmdd]
    log_path.write_text(header + ''.join(kept))


def run_commands(gcd: Path, yymm: str):
    """Same pipeline as m4_ingest.py, using the reimported month's yymm and a
    wide sample-log merge duration so historical directories are covered."""
    cmds = [
        ["/hats/gc/gcwerks-3/bin/gcimport",    "-gcdir", str(gcd)],
        ["/hats/gc/gcwerks-3/bin/run-index",   "-gcdir", str(gcd)],
        ["/hats/gc/itxbin/m4_samplogs.py",     "--all", "-i"],
        ["/hats/gc/gcwerks-3/bin/gcupdate",    "-gcdir", str(gcd), yymm],
        ["/hats/gc/gcwerks-3/bin/gccalc",      "-gcdir", str(gcd)],
        ["/hats/gc/itxbin/m4_gcwerks2db.py",   yymm, "-x", "--flagged"],
        ["/hats/gc/itxbin/m4_batch.py",        "-p", "all", "-i"],
    ]
    for cmd in cmds:
        logging.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Remove and reimport a single M4 chemstation directory."
    )
    parser.add_argument("dirname", help="Chemstation dir name or path, e.g. bd062626")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip the confirmation prompt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    name = Path(args.dirname).name
    chemstation_path = CHEMSTATION_DIR / name
    date_obj = parse_bd_date(name)
    if date_obj is None:
        parser.error(f"Could not parse a bdMMDDYY date from '{name}'")
    if not chemstation_path.is_dir():
        parser.error(f"Chemstation directory not found: {chemstation_path}")

    yy = str(date_obj.year)[2:]
    yymmdd = date_obj.strftime('%y%m%d')

    chrom_files = find_chromatograms(yy, yymmdd)
    kept_lines, removed_lines = find_runindex_lines(yymmdd)
    samplelog_path = find_samplelog_rows(date_obj, yymmdd)
    samplelog_count = count_samplelog_rows(samplelog_path, yymmdd) if samplelog_path else 0

    print(f"Reimporting {name} (date {date_obj.isoformat()}):")
    print(f"  Chemstation dir:        {chemstation_path}")
    print(f"  Chromatograms to remove: {len(chrom_files)} in {GC_DIR / yy / 'chromatograms' / 'channel0'}")
    print(f"  .run-index lines to remove: {len(removed_lines)}")
    if samplelog_path:
        print(f"  sample.log rows to remove: {samplelog_count} in {samplelog_path}")
    else:
        print("  sample.log file for this year not found (nothing to remove there)")

    if not removed_lines and not chrom_files and not samplelog_count:
        print("Nothing found to clean up for this date -- proceeding to reimport only.")

    if not args.yes:
        answer = input("Proceed with removal and reimport? [y/N] ").strip().lower()
        if answer != 'y':
            print("Aborted.")
            return

    for f in chrom_files:
        f.unlink()
        logging.info(f"Removed chromatogram: {f}")

    if removed_lines:
        runindex = GC_DIR / '.run-index'
        runindex.write_text(''.join(kept_lines))
        logging.info(f"Removed {len(removed_lines)} lines from {runindex}")

    if samplelog_path and samplelog_count:
        remove_samplelog_rows(samplelog_path, yymmdd)
        logging.info(f"Removed {samplelog_count} rows from {samplelog_path}")

    yymm = date_obj.strftime('%y%m')
    run_commands(GC_DIR, yymm)


if __name__ == "__main__":
    main()
