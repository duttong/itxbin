#!/usr/bin/env python3

import subprocess
import sys
from datetime import date
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Ingest and process IE3 GC data.")


@app.command()
def ingest(
    site: str = typer.Option("smo", "--site", help="Station code for the IE3 instrument."),
    all_data: bool = typer.Option(False, "--all", help="Process all data (overrides --year)."),
    year: int = typer.Option(date.today().year, "--year", help="Process a single year (YYYY)."),
):
    """Run IE3 export followed by the database load process."""
    bin_dir = Path(__file__).resolve().parent

    # 1. Run ie3_export.py
    export_script = str(bin_dir / "ie3_export.py")
    export_cmd = [export_script, "all", "--site", site]
    if not all_data:
        export_cmd.extend(["--year", str(year)])
        
    typer.secho(f"Running: {' '.join(export_cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(export_cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Error running ie3_export.py: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # 2. Run ie3_gcwerks2db.py
    load_script = str(bin_dir / "ie3_gcwerks2db.py")
    load_cmd = [load_script, site]
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
        
    typer.secho("Ingest complete.", fg=typer.colors.GREEN, bold=True)

if __name__ == "__main__":
    app()