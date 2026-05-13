#!/usr/bin/env python3
"""
Audit archived sample sheets for a site against hats.Status_MetData.

Reports PairIDs present in the database that have no corresponding archived
PDF in /hats/gc/sample_sheets/archived/{site}/.
"""

import csv
import re
import sys
from pathlib import Path

import typer
from typing_extensions import Annotated

sys.path.append('/ccg/src/db/')
import db_utils.db_conn as db_conn

ARCHIVE_ROOT = Path("/hats/gc/sample_sheets/archived")
FILENAME_RE = re.compile(r"^logos_[a-z]+_(\d+)_", re.IGNORECASE)

app = typer.Typer(help=__doc__)


def archived_pair_ids(site: str) -> set[int]:
    """Return the set of PairIDs found in archived PDF filenames for the site."""
    site_dir = ARCHIVE_ROOT / site.lower()
    if not site_dir.is_dir():
        typer.echo(f"Archive directory not found: {site_dir}", err=True)
        raise typer.Exit(1)

    ids: set[int] = set()
    for path in site_dir.iterdir():
        if not path.is_file():
            continue
        m = FILENAME_RE.match(path.name)
        if m:
            ids.add(int(m.group(1)))
    return ids


def db_records(site: str) -> dict[int, str | None]:
    """Return {PairID: sample_datetime_utc} for all DB records at the station."""
    db = db_conn.HATS_ng()
    sql = "SELECT PairID, sample_datetime_utc FROM hats.Status_MetData WHERE Station = %s"
    rows = db.doquery(sql, [site.upper()])
    return {int(r["PairID"]): r["sample_datetime_utc"] for r in rows}


@app.command()
def main(
    site: Annotated[str, typer.Argument(help="Site code (e.g. brw, mlo, spo).")],
    sort_datetime: Annotated[
        bool,
        typer.Option("--sort-datetime", help="Sort output by sample_datetime instead of pairid."),
    ] = False,
    orphans: Annotated[
        bool,
        typer.Option("--orphans", help="Also report archived sheets with no matching DB record."),
    ] = False,
):
    archived = archived_pair_ids(site)
    records = db_records(site)

    if not records:
        typer.echo(f"No records found in Status_MetData for station {site.upper()!r}.", err=True)
        raise typer.Exit(1)

    writer = csv.writer(sys.stdout)
    writer.writerow(["pairid", "sample_datetime", "site"])

    if orphans:
        orphan_ids = sorted(archived - records.keys())
        for pid in orphan_ids:
            writer.writerow([pid, "", site.upper()])
        typer.echo(
            f"# {len(orphan_ids)} orphaned of {len(archived)} archived sheets "
            f"({len(records)} DB records)",
            err=True,
        )
    else:
        missing = [pid for pid in records if pid not in archived]
        if sort_datetime:
            missing.sort(key=lambda pid: (records[pid] is None, records[pid]))
        else:
            missing.sort()
        for pid in missing:
            dt = records[pid]
            writer.writerow([pid, dt if dt is not None else "", site.upper()])
        typer.echo(
            f"# {len(missing)} missing of {len(records)} DB records "
            f"({len(archived)} archived sheets)",
            err=True,
        )


if __name__ == "__main__":
    app()
