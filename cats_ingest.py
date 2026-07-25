#!/usr/bin/env python3
"""Ingest CATS GC data: export from GCwerks, load into DB, recompute mole
fractions, optionally import legacy published mole fractions."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(add_completion=False, help="Ingest CATS GC data into the HATS database.")

BIN_DIR = Path(__file__).resolve().parent
CATS_EXPORT = Path("/home/hats/gdutton/bin/cats_export.py")

# Must match CATS_Instrument.start_date in logosdata/logos_instruments_insitu.py
CATS_START_DATE = "1998-01-01"
# Matches the duration_months default in cats_gcwerks2db.py / cats_aftp2db.py,
# so the whole pipeline shares one default lookback window.
MF_LOOKBACK_MONTHS = 2


def _run(cmd: list[str]):
    label = " ".join(cmd)
    typer.secho(f"Running: {label}", fg=typer.colors.BLUE)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def _resolve_mf_window(all_data: bool, year: int | None) -> tuple[str, str | None]:
    """Resolve (start_date, end_date) for the mf-calc steps (step 4), so
    cats_set_mf_method.py and cats_batch.py process the same window --
    a row is only ever tagged cal12/cal1 in the same run that recomputes it.
    """
    if all_data:
        return CATS_START_DATE, None
    if year is not None:
        return f"{year}-01-01", f"{year}-12-31"
    start = pd.Timestamp.now() - pd.DateOffset(months=MF_LOOKBACK_MONTHS)
    return start.strftime("%Y-%m-%d"), None


@app.command()
def ingest(
    site: str = typer.Argument(..., help="Site code: brw, spo"),
    all_data: bool = typer.Option(False, "--all", help="Process all data (overrides --year)"),
    year: int | None = typer.Option(None, "--year", help="Process a single year (YYYY)"),
    flagged: bool = typer.Option(True, "--flagged/--no-flagged", help="Parse and sync GCwerks flag characters"),
    skip_export: bool = typer.Option(False, "--skip-export", help="Skip GCwerks CSV export step"),
    run_aftp: bool = typer.Option(
        False, "--run-aftp", help="Also import legacy published mole fractions from /aftp (one-time backfill; off by default)"
    ),
    mf: bool = typer.Option(
        True, "--mf/--no-mf",
        help="Assign default cal methods and recompute weekly cal fits + "
             "mole fractions via cats_set_mf_method.py/cats_batch.py (default: on)",
    ),
):
    """Ingest CATS data for one site.

    Steps:
      1. Export GCwerks data to /hats/gc/cats_results/ (cats_export.py)
      2. Load GCwerks CSVs into ng_insitu_analysis / ng_insitu_mole_fractions
      3. (optional, --run-aftp) Import legacy published mole fractions from /aftp/hats
      4. (default, --mf) Assign default calibration methods, then recompute
         weekly cal fits and mole fractions (cats_set_mf_method.py, cats_batch.py)
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

    # 3. Import legacy published mole fractions from /aftp/hats (one-time backfill, opt-in)
    if run_aftp:
        aftp_cmd = [sys.executable, str(BIN_DIR / "cats_aftp2db.py"), site]
        if all_data:
            aftp_cmd.append("--all")
        elif year is not None:
            aftp_cmd.extend(["--year", str(year)])
        _run(aftp_cmd)

    # 4. Recompute mole fractions: assign default cal methods, then weekly
    # fits + mole fractions. Both steps share the same date window so a row
    # is only ever tagged cal12/cal1 in the same run that recomputes it.
    if mf:
        start_date, end_date = _resolve_mf_window(all_data, year)

        method_cmd = [
            sys.executable, str(BIN_DIR / "cats_set_mf_method.py"),
            "--site", site, "--start-date", start_date,
        ]
        _run(method_cmd)

        batch_cmd = [
            sys.executable, str(BIN_DIR / "cats_batch.py"),
            "--site", site, "-p", "all", "-i", "--fits", "-s", start_date,
        ]
        if end_date:
            batch_cmd.extend(["-e", end_date])
        _run(batch_cmd)

    typer.secho(f"CATS ingest complete for {site.upper()}.", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
