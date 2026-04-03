#!/usr/bin/env python3

import shutil
import subprocess
from datetime import date
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Ingest and process IE3 GC data.")


@app.command()
def ingest(
    site: str = typer.Option("smo", "--site", help="Station code for the IE3 instrument."),
    all_data: bool = typer.Option(False, "--all", help="Process all data (overrides --year)."),
    year: int = typer.Option(date.today().year, "--year", help="Process a single year (YYYY)."),
    past_days: int = typer.Option(2, "--past-days", help="Number of recent days to import (default: 2)."),
):
    """Import IE3 chromatograms then load results into the database."""
    bin_dir = Path(__file__).resolve().parent

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

    # 3. Export from GCwerks and load into database (ie3_export runs automatically inside)
    load_cmd = [str(bin_dir / "ie3_gcwerks2db.py"), site, "--flagged"]
    if all_data:
        load_cmd.append("--all")
    else:
        load_cmd.extend(["--year", str(year)])

    typer.secho(f"Running: {' '.join(load_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(load_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_gcwerks2db.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 4. Upsert sample loop temp/pressure/flow from engineering data
    eng_cmd = [str(bin_dir / "ie3_eng2db.py"), "--site", site]
    if all_data:
        eng_cmd.append("--all")
    else:
        eng_cmd.extend(["--year", str(year)])
    typer.secho(f"Running: {' '.join(eng_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(eng_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_eng2db.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 5. Recalculate and write mole fractions for all analytes
    batch_cmd = [str(bin_dir / "ie3_batch.py"), "--site", site, "-p", "all", "-i"]
    typer.secho(f"Running: {' '.join(batch_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(batch_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_batch.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("Ingest complete.", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
