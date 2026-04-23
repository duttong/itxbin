#!/usr/bin/env python3
"""Split multi-page PDFs in a directory into single-page files.

Mirrors split_multipage_pdfs.sh: finds *.pdf in --dir, splits any with
>1 page via pdfseparate, moves originals into a 'converted/' subdir,
and optionally calls `checkin process` on the resulting pages.
"""

import argparse
import fcntl
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SPLIT_SUFFIX_RE = re.compile(r"_\d{3}\.pdf$", re.IGNORECASE)
CHECKIN_SENTINEL = "CHECKIN_ENABLED"


def pdf_page_count(pdf: Path) -> int | None:
    """Return page count from pdfinfo, or None if unreadable."""
    try:
        out = subprocess.run(
            ["pdfinfo", str(pdf)],
            check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def split_pdf(pdf: Path, dry_run: bool) -> list[Path]:
    """Run pdfseparate on pdf; return the list of output page paths."""
    stem = pdf.with_suffix("")
    pattern = f"{stem}_%03d.pdf"
    if not dry_run:
        subprocess.run(["pdfseparate", str(pdf), pattern], check=True)
    pages = pdf_page_count(pdf) or 0
    return [Path(f"{stem}_{i:03d}.pdf") for i in range(1, pages + 1)]


def process_once(
    work_dir: Path,
    converted_dir: Path,
    dry_run: bool,
    exclude: set[Path],
) -> tuple[list[Path], list[Path]]:
    """One sweep. Returns (newly_split_pages, pass_through_pdfs).

    pass_through_pdfs are files ready for checkin as-is: single-page PDFs
    and any pre-existing `_NNN.pdf` outputs. `exclude` holds resolved paths
    already handed to checkin in a prior sweep so we don't resubmit them.
    """
    split_outputs: list[Path] = []
    pass_through: list[Path] = []
    for pdf in sorted(work_dir.glob("*.pdf")) + sorted(work_dir.glob("*.PDF")):
        if pdf.resolve() in exclude:
            continue
        if SPLIT_SUFFIX_RE.search(pdf.name):
            pass_through.append(pdf)
            continue
        pages = pdf_page_count(pdf)
        if pages is None:
            print(f"Skipping unreadable PDF: {pdf}", file=sys.stderr)
            continue
        if pages == 1:
            pass_through.append(pdf)
            continue
        print(f"Splitting: {pdf} ({pages} pages)")
        split_outputs.extend(split_pdf(pdf, dry_run))
        dest = converted_dir / pdf.name
        if dry_run:
            print(f"[dry-run] Would move original to: {dest}")
        else:
            shutil.move(str(pdf), dest)
            print(f"Moved original to: {dest}")
    return split_outputs, pass_through


def run_checkin(
    checkin_cmd: Path,
    pages: list[Path],
    work_dir: Path,
    dry_run: bool,
    log_path: Path,
) -> None:
    names = [p.name for p in pages]
    cmd = [str(checkin_cmd), "process", *names]
    if dry_run:
        print(f"[dry-run] Would run: {' '.join(cmd)} (cwd={work_dir})")
        print(f"[dry-run] Would append output to: {log_path}")
        return

    start = datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    end = datetime.now().isoformat(timespec="seconds")

    with log_path.open("a") as f:
        f.write(f"\n=== {start} checkin process ({len(names)} file(s)) cwd={work_dir} ===\n")
        f.write(f"cmd: {' '.join(cmd)}\n")
        if result.stdout:
            f.write(result.stdout)
            if not result.stdout.endswith("\n"):
                f.write("\n")
        if result.stderr:
            f.write("[stderr]\n")
            f.write(result.stderr)
            if not result.stderr.endswith("\n"):
                f.write("\n")
        f.write(f"--- {end} exit={result.returncode} ---\n")

    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr,
        )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-d", "--dir", type=Path, default=Path.cwd(),
                        help="directory to scan (default: cwd)")
    parser.add_argument("--converted-dir", type=Path, default=None,
                        help="where to move originals (default: <dir>/converted)")
    parser.add_argument("--checkin-cmd", type=Path, default=script_dir / "checkin",
                        help="path to the checkin command")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--checkin", dest="checkin", action="store_const", const=True,
                     help="force checkin on (overrides sentinel)")
    grp.add_argument("--no-checkin", dest="checkin", action="store_const", const=False,
                     help="force checkin off (overrides sentinel)")
    parser.set_defaults(checkin=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print actions without modifying files")
    parser.add_argument("--lock-file", type=Path, default=None,
                        help="lock file path (default: <dir>/.split_multipage_pdfs.lock)")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="checkin output log (default: <dir>/checkin.log)")
    args = parser.parse_args()

    work_dir: Path = args.dir.resolve()
    if not work_dir.is_dir():
        parser.error(f"not a directory: {work_dir}")
    converted_dir: Path = (args.converted_dir or work_dir / "converted").resolve()
    lock_path: Path = (args.lock_file or work_dir / ".split_multipage_pdfs.lock").resolve()
    log_path: Path = (args.log_file or work_dir / "checkin.log").resolve()

    if not args.dry_run:
        converted_dir.mkdir(parents=True, exist_ok=True)

    sentinel_path = work_dir / CHECKIN_SENTINEL
    if args.checkin is None:
        checkin_on = sentinel_path.exists()
        source = f"sentinel {'present' if checkin_on else 'absent'} ({sentinel_path.name})"
    else:
        checkin_on = args.checkin
        source = "CLI flag"
    print(f"Checkin mode: {'ON' if checkin_on else 'OFF'} (from {source})")

    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("split_multipage_pdfs is already running; exiting.")
        return 0

    submitted: set[Path] = set()
    processed_any = False
    while True:
        split_outputs, pass_through = process_once(
            work_dir, converted_dir, args.dry_run, submitted,
        )
        if checkin_on:
            batch = split_outputs + pass_through
            if not batch:
                break
            processed_any = True
            print(f"Checkin: {len(batch)} PDF(s) "
                  f"({len(split_outputs)} split, {len(pass_through)} as-is).")
            run_checkin(args.checkin_cmd, batch, work_dir, args.dry_run, log_path)
            submitted.update(p.resolve() for p in batch)
        else:
            if split_outputs:
                processed_any = True
                print(f"Split {len(split_outputs)} PDF page(s); skipping checkin.")
            break

    if not processed_any:
        print("No multi-page PDFs found to split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
