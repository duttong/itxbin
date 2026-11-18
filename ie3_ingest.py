#!/usr/bin/env python3

import shutil
import subprocess
from pathlib import Path

import typer

import logos_dates

app = typer.Typer(add_completion=False, help="Ingest and process IE3 GC data.")


@app.command()
def ingest(
    site: str = typer.Option("smo", "--site", help="Station code for the IE3 instrument."),
    start: str | None = typer.Option(None, "-s", "--start", help=f"first time to load: {logos_dates.DATE_HELP} (default: start of last month)"),
    end: str | None = typer.Option(None, "-e", "--end", help=f"last time to load, inclusive: {logos_dates.DATE_HELP} (default: now)"),
    all_data: bool = typer.Option(False, "--all", help="Process all data."),
    year: int | None = typer.Option(None, "--year", help="Process a single year (YYYY)."),
    past_days: int = typer.Option(2, "--past-days", help="Number of recent days to import (default: 2)."),
):
    """Import IE3 chromatograms then load results into the database.

    The range (-s/-e, --year or --all; default from the start of last month to now) goes to the
    loader, ie3_eng2db.py and ie3_batch.py, so they cover the same analyses. --past-days is only the
    chromatogram import."""
    bin_dir = Path(__file__).resolve().parent
    try:
        t0, t1 = logos_dates.resolve(start=start, end=end, all_data=all_data, year=year)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    range_args = ["--all"] if all_data else logos_dates.range_cli_args(t0, t1)

    # 1. Import chromatograms from incoming tarballs into GCwerks
    import_cmd = [str(bin_dir / "ie3_import.py"), "--past-days", str(past_days), site]
    typer.secho(f"Running: {' '.join(import_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(import_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_import.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 2. Copy *.log and *.csv from incoming to /hats/gc/{site}/logs
    incoming = Path(f"/nfs/isftp/sftp/data/logos/{site}/incoming")
    logs_dest = Path(f"/hats/gc/{site}/logs")
    for pattern in ("*.log", "*.csv"):
        for src in incoming.glob(pattern):
            try:
                shutil.copy2(src, logs_dest / src.name)
            except Exception as e:
                typer.secho(f"Warning: could not copy {src.name}: {e}", fg=typer.colors.YELLOW, err=True)

    # 2b. Load the operator cylinder-pressure log into hats.ng_cylinder_pressures
    # (read by the omi field-cylinders dashboard). Warn only; it must not block
    # GC data processing.
    cyl_cmd = [str(bin_dir / "ie3_cylinders2db.py"), "--site", site]
    typer.secho(f"Running: {' '.join(cyl_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(cyl_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Warning: ie3_cylinders2db.py failed: {e}", fg=typer.colors.YELLOW, err=True)

    # 3. Export from GCwerks and load into database. The --flagged path writes
    # GCwerks flags to hats.ng_insitu_mole_fraction_tags.
    load_cmd = [str(bin_dir / "ie3_gcwerks2db.py"), site, "--flagged", *range_args]

    typer.secho(f"Running: {' '.join(load_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(load_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_gcwerks2db.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 4. Upsert sample loop temp/pressure/flow from engineering data
    eng_cmd = [str(bin_dir / "ie3_eng2db.py"), "--site", site, *range_args]
    typer.secho(f"Running: {' '.join(eng_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(eng_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_eng2db.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 5. Set the calibration method (cal12/cal1) on any newly-ingested rows
    # before computing fits, so update_fits() builds the right kind of fit.
    method_cmd = [str(bin_dir / "ie3_set_mf_method.py")]
    typer.secho(f"Running: {' '.join(method_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(method_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_set_mf_method.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 5b. Response corrections for the AC-cycle temperature effect (temporary
    # SMO location, 2026-05-11 on) for rows that don't have one yet, so
    # ie3_batch.py below applies them. Warn only: missing factors mean
    # uncorrected (factor 1), not wrong.
    corr_cmd = [str(bin_dir / "ie3_resp_correction.py"), "--new-only", "-i"]
    typer.secho(f"Running: {' '.join(corr_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(corr_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Warning: ie3_resp_correction.py failed: {e}", fg=typer.colors.YELLOW, err=True)

    # 6. Recalculate and write mole fractions for all analytes
    batch_cmd = [str(bin_dir / "ie3_batch.py"), "--site", site, "-p", "all", "-i", "--fits", *range_args]
    typer.secho(f"Running: {' '.join(batch_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(batch_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_batch.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("Ingest complete.", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
