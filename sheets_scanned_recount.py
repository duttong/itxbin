#!/usr/bin/env python3
"""
Recount the archived sample-sheet scans of every pair and store the count in
the pair tables' `scanned` column (0, 1, 2, ...):

  hats.hatsflask_pair_info  <- /nfs/hats/gc/sample_sheets/archived/<site>/
                               logos_<site>_<pairid>[n<k>]_<YYYY.MM.DD_HH.MM.SS>.<ext>
  ccgg.flask_pair_info      <- /nfs/ccg/network/sample_sheets/flask/surface_v2/<site>/
                               flask_surface_<site>_<pairid 6-digit>_<stamp>[_<k>].<ext>

The check-in pages and ccg_sheet_archive.py set `scanned` when they archive a
scan; this is the backfill and a nightly check that catches anything else
(a sheet archived by hand or by the desktop checkin GUI, a renamed copy).

A file is counted for a pair only when the pair is at that folder's site, so a
legacy id in an old file name never credits another site's pair.

Dry run by default: prints what would change.  --apply writes the changes.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.extend(["/ccg/src/db", "/ccg/src/db/db_utils"])
import db_conn  # noqa: E402

HATS_DIR = Path("/nfs/hats/gc/sample_sheets/archived")
CCG_DIR = Path("/nfs/ccg/network/sample_sheets/flask/surface_v2")
EXT = {".pdf", ".jpg", ".jpeg"}
MAX_SCANNED = 127                      # tinyint


def count_files(root: Path, pattern: str) -> Counter:
    """(site lower, pair id) -> number of scans, from one listing per site folder."""
    out = Counter()
    for d in sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"[a-z0-9]{3}", p.name)):
        rx = re.compile(pattern.format(site=re.escape(d.name)))
        for f in d.iterdir():
            if f.suffix.lower() not in EXT:
                continue
            m = rx.match(f.name)
            if m and int(m.group(1)) > 0:
                out[(d.name, int(m.group(1)))] += 1
    return out


def recount(db, table: str, site_sql: str, files: Counter, apply: bool, label: str) -> None:
    sites = {int(r["pair_id"]): (r["code"] or "").lower() for r in db.doquery(site_sql) or []}
    stored = {int(r["pair_id"]): int(r["scanned"] or 0)
              for r in db.doquery(f"SELECT pair_id, scanned FROM {table}") or []}
    want = {pid: 0 for pid in stored}
    other_site = 0
    for (site, pid), n in files.items():
        if pid not in stored:
            continue
        if sites.get(pid) != site:
            other_site += n
            continue
        want[pid] = min(MAX_SCANNED, want[pid] + n)
    change = {pid: n for pid, n in want.items() if n != stored[pid]}
    dist = Counter(want.values())
    print(f"{label}: {len(stored)} pairs; scanned counts {dict(sorted(dist.items()))}; "
          f"{len(change)} to change; {other_site} file(s) skipped (pair at another site or legacy id)")
    for pid in sorted(change)[:10]:
        print(f"  pair {pid}: {stored[pid]} -> {change[pid]}")
    if len(change) > 10:
        print(f"  ... {len(change) - 10} more")
    if not apply or not change:
        return
    by_n = {}
    for pid, n in change.items():
        by_n.setdefault(n, []).append(pid)
    done = 0
    for n, pids in sorted(by_n.items()):
        for i in range(0, len(pids), 1000):
            chunk = pids[i:i + 1000]
            db.doquery(f"UPDATE {table} SET scanned = %s WHERE pair_id IN ({','.join(['%s'] * len(chunk))})",
                       [n] + chunk)
            done += len(chunk)
    print(f"  updated {done} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the changed counts (default: dry run)")
    ap.add_argument("--only", choices=["hats", "ccg"], help="one project only")
    args = ap.parse_args()
    db = db_conn.HATS_ng()
    if args.only in (None, "hats"):
        files = count_files(HATS_DIR, r"^logos_{site}_(\d+)(?:n\d+)?_")
        recount(db, "hats.hatsflask_pair_info",
                "SELECT p.pair_id, s.code FROM hats.hatsflask_pair_info p JOIN gmd.site s ON s.num = p.site_num",
                files, args.apply, "HATS Flask")
    if args.only in (None, "ccg"):
        files = count_files(CCG_DIR, r"^flask_surface_{site}_(\d{{6}})_")
        # ccgg.flask_pair_info has no site: the site of the pair's events
        recount(db, "ccgg.flask_pair_info",
                "SELECT e.pair_id, MIN(s.code) code FROM ccgg.flask_event e JOIN gmd.site s ON s.num = e.site_num "
                "WHERE e.pair_id > 0 GROUP BY e.pair_id",
                files, args.apply, "CCG Surface")


if __name__ == "__main__":
    main()
