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
from pathlib import Path


SPLIT_SUFFIX_RE = re.compile(r"_\d{3}\.pdf$", re.IGNORECASE)


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


def process_once(work_dir: Path, converted_dir: Path, dry_run: bool) -> list[Path]:
    """One sweep: split all multi-page PDFs in work_dir. Returns produced pages."""
    produced: list[Path] = []
    for pdf in sorted(work_dir.glob("*.pdf")) + sorted(work_dir.glob("*.PDF")):
        if SPLIT_SUFFIX_RE.search(pdf.name):
            continue
        pages = pdf_page_count(pdf)
        if pages is None:
            print(f"Skipping unreadable PDF: {pdf}", file=sys.stderr)
            continue
        if pages <= 1:
            print(f"Skipping single-page PDF: {pdf}")
            continue
        print(f"Splitting: {pdf} ({pages} pages)")
        produced.extend(split_pdf(pdf, dry_run))
        dest = converted_dir / pdf.name
        if dry_run:
            print(f"[dry-run] Would move original to: {dest}")
        else:
            shutil.move(str(pdf), dest)
            print(f"Moved original to: {dest}")
    return produced


def run_checkin(checkin_cmd: Path, pages: list[Path], work_dir: Path, dry_run: bool) -> None:
    names = [p.name for p in pages]
    cmd = [str(checkin_cmd), "process", *names]
    if dry_run:
        print(f"[dry-run] Would run: {' '.join(cmd)} (cwd={work_dir})")
        return
    subprocess.run(cmd, check=True, cwd=work_dir)


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
    grp.add_argument("--checkin", dest="checkin", action="store_true",
                     help="call `checkin process` on split pages")
    grp.add_argument("--no-checkin", dest="checkin", action="store_false",
                     help="skip checkin invocation (default)")
    parser.set_defaults(checkin=False)
    parser.add_argument("--dry-run", action="store_true",
                        help="print actions without modifying files")
    parser.add_argument("--lock-file", type=Path, default=None,
                        help="lock file path (default: <dir>/.split_multipage_pdfs.lock)")
    args = parser.parse_args()

    work_dir: Path = args.dir.resolve()
    if not work_dir.is_dir():
        parser.error(f"not a directory: {work_dir}")
    converted_dir: Path = (args.converted_dir or work_dir / "converted").resolve()
    lock_path: Path = (args.lock_file or work_dir / ".split_multipage_pdfs.lock").resolve()

    if not args.dry_run:
        converted_dir.mkdir(parents=True, exist_ok=True)

    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("split_multipage_pdfs is already running; exiting.")
        return 0

    processed_any = False
    while True:
        pages = process_once(work_dir, converted_dir, args.dry_run)
        if not pages:
            break
        processed_any = True
        if args.checkin:
            print(f"Processing {len(pages)} split PDF page(s).")
            run_checkin(args.checkin_cmd, pages, work_dir, args.dry_run)
        else:
            print(f"Split {len(pages)} PDF page(s); skipping checkin.")
            break

    if not processed_any:
        print("No multi-page PDFs found to split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
