#!/usr/bin/env python3
"""
Audit archived sample sheets against hats.Status_MetData.

Reports PairIDs present in the database that have no corresponding archived
PDF in /hats/gc/sample_sheets/archived/{site}/. A single site or all archived
sites can be checked.
"""

import csv
import re
import sys
from pathlib import Path
from typing import Optional

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


def archived_sites() -> list[str]:
    """Return three-letter site codes represented by archive directories."""
    return sorted(
        path.name.upper()
        for path in ARCHIVE_ROOT.iterdir()
        if path.is_dir() and len(path.name) == 3 and path.name.isalpha()
    )


def db_records(site: Optional[str] = None) -> dict[int, tuple[str | None, str]]:
    """Return {PairID: (sample_datetime_utc, station)} for DB records."""
    db = db_conn.HATS_ng()
    sql = "SELECT PairID, sample_datetime_utc, Station FROM hats.Status_MetData"
    params = None
    if site is not None:
        sql += " WHERE Station = %s"
        params = [site.upper()]
    rows = db.doquery(sql, params)
    return {
        int(r["PairID"]): (r["sample_datetime_utc"], r["Station"].upper())
        for r in rows
    }


@app.command()
def main(
    site: Annotated[
        Optional[str],
        typer.Argument(help="Site code (e.g. brw, mlo, spo). Omit with --all."),
    ] = None,
    all_sites: Annotated[
        bool,
        typer.Option("--all", help="Return combined results for all archived sites."),
    ] = False,
    sort_datetime: Annotated[
        bool,
        typer.Option("--sort-datetime", help="Sort output by sample_datetime instead of pairid."),
    ] = False,
    orphans: Annotated[
        bool,
        typer.Option("--orphans", help="Also report archived sheets with no matching DB record."),
    ] = False,
):
    if all_sites and site is not None:
        typer.echo("Specify either a site or --all, not both.", err=True)
        raise typer.Exit(2)
    if not all_sites and site is None:
        typer.echo("Specify a site or use --all.", err=True)
        raise typer.Exit(2)

    sites = archived_sites() if all_sites else [site.upper()]
    archived_by_site = {
        station: archived_pair_ids(station)
        for station in sites
    }
    records = db_records(None if all_sites else site)
    if all_sites:
        records = {
            pid: record
            for pid, record in records.items()
            if record[1] in archived_by_site
        }

    if not records:
        target = "archived sites" if all_sites else f"station {site.upper()!r}"
        typer.echo(f"No records found in Status_MetData for {target}.", err=True)
        raise typer.Exit(1)

    writer = csv.writer(sys.stdout)
    writer.writerow(["pairid", "sample_datetime", "site"])

    if orphans:
        record_ids_by_site = {station: set() for station in sites}
        for pid, (_, station) in records.items():
            record_ids_by_site[station].add(pid)
        orphan_rows = sorted(
            (pid, station)
            for station, archived in archived_by_site.items()
            for pid in archived - record_ids_by_site[station]
        )
        for pid, station in orphan_rows:
            writer.writerow([pid, "", station])
        archived_count = sum(len(ids) for ids in archived_by_site.values())
        typer.echo(
            f"# {len(orphan_rows)} orphaned of {archived_count} archived sheets "
            f"({len(records)} DB records)",
            err=True,
        )
    else:
        missing = [
            (pid, dt, station)
            for pid, (dt, station) in records.items()
            if pid not in archived_by_site[station]
        ]
        if sort_datetime:
            missing.sort(key=lambda row: (row[1] is None, row[1]))
        else:
            missing.sort(key=lambda row: row[0])
        for pid, dt, station in missing:
            writer.writerow([pid, dt if dt is not None else "", station])
        archived_count = sum(len(ids) for ids in archived_by_site.values())
        typer.echo(
            f"# {len(missing)} missing of {len(records)} DB records "
            f"({archived_count} archived sheets)",
            err=True,
        )


if __name__ == "__main__":
    app()
