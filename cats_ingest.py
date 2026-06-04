#!/usr/bin/env python3
"""Ingest CATS GC data: export from GCwerks, load into DB, import published mole fractions."""

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Ingest CATS GC data into the HATS database.")

BIN_DIR = Path(__file__).resolve().parent
CATS_EXPORT = Path("/home/hats/gdutton/bin/cats_export.py")


def _run(cmd: list[str]):
    label = " ".join(cmd)
    typer.secho(f"Running: {label}", fg=typer.colors.BLUE)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def ingest(
    site: str = typer.Argument(..., help="Site code: brw, spo"),
    all_data: bool = typer.Option(False, "--all", help="Process all data (overrides --year)"),
    year: int | None = typer.Option(None, "--year", help="Process a single year (YYYY)"),
    flagged: bool = typer.Option(True, "--flagged/--no-flagged", help="Parse and sync GCwerks flag characters"),
    skip_export: bool = typer.Option(False, "--skip-export", help="Skip GCwerks CSV export step"),
    skip_aftp: bool = typer.Option(False, "--skip-aftp", help="Skip /aftp mole fraction import"),
):
    """Ingest CATS data for one site.

    Steps:
      1. Export GCwerks data to /hats/gc/cats_results/ (cats_export.py)
      2. Load GCwerks CSVs into ng_insitu_analysis / ng_insitu_mole_fractions
      3. Import published mole fractions from /aftp/hats into ng_insitu_mole_fractions
    """
    # 1. GCwerks export
    if not skip_export:
        cmd = [sys.executable, str(CATS_EXPORT), site]
        if flagged:
            cmd.append("--flagged")
        _run(cmd)

    # 2. Load GCwerks results into DB
    gcwerks_cmd = [sys.executable, str(BIN_DIR / "cats_gcwerks2db.py"), site]
    if flagged:
        gcwerks_cmd.append("--flagged")
    if all_data:
        gcwerks_cmd.append("--all")
    elif year is not None:
        gcwerks_cmd.extend(["--year", str(year)])
    _run(gcwerks_cmd)

    # 3. Import published mole fractions from /aftp/hats
    if not skip_aftp:
        aftp_cmd = [sys.executable, str(BIN_DIR / "cats_aftp2db.py"), site]
        if all_data:
            aftp_cmd.append("--all")
        elif year is not None:
            aftp_cmd.extend(["--year", str(year)])
        _run(aftp_cmd)

    typer.secho(f"CATS ingest complete for {site.upper()}.", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
