#!/usr/bin/env python3

import shutil
import subprocess
from pathlib import Path

import typer

import ingest_common
import logos_dates
import logos_lock

app = typer.Typer(add_completion=False, help="Ingest and process IE3 GC data.")


STEPS = ("import", "update", "load", "eng", "methods", "batch")
DAILY = ("import", "load", "eng", "methods", "batch")    # ie3_import.py runs gcupdate/gccalc itself when it imports
GCWERKS = "/hats/gc/gcwerks-3/bin"


def _run(cmd, warn=False):
    """Run one step's command; a failure stops the ingest, or with warn=True only warns."""
    typer.secho(f"Running: {' '.join(cmd)}", fg=typer.colors.BLUE)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        name = Path(cmd[0]).name
        if warn:
            typer.secho(f"Warning: {name} failed: {e}", fg=typer.colors.YELLOW, err=True)
            return
        typer.secho(f"Error running {name}: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def ingest(
    site: str = typer.Option("smo", "--site", help="Station code for the IE3 instrument."),
    start: str | None = typer.Option(None, "-s", "--start", help=f"first time to load: {logos_dates.DATE_HELP} (default: start of last month)"),
    end: str | None = typer.Option(None, "-e", "--end", help=f"last time to load, inclusive: {logos_dates.DATE_HELP} (default: now)"),
    all_data: bool = typer.Option(False, "--all", help="Process all data."),
    year: int | None = typer.Option(None, "--year", help="Process a single year (YYYY)."),
    past_days: int = typer.Option(2, "--past-days", help="Number of recent days to import (default: 2)."),
    steps_opt: str = typer.Option(",".join(DAILY), "--steps",
                                  help=f"steps to run, comma separated, from: {', '.join(STEPS)}"),
):
    """Import IE3 chromatograms then load results into the database.

    The range (-s/-e, --year or --all; default from the start of last month to now) goes to the
    loader, ie3_eng2db.py and ie3_batch.py, so they cover the same analyses. --past-days is only the
    chromatogram import. --steps picks steps (update = gcupdate from the range's month + gccalc, off
    by default). The run holds logos_lock 'ie3-<site>'."""
    bin_dir = Path(__file__).resolve().parent
    try:
        t0, t1 = logos_dates.resolve(start=start, end=end, all_data=all_data, year=year)
        steps = ingest_common.parse_steps(steps_opt, STEPS)
    except ValueError as e:
        raise typer.BadParameter(str(e))
    range_args = ["--all"] if all_data else logos_dates.range_cli_args(t0, t1)
    typer.secho(f"IE3 {site.upper()} ingest: steps {','.join(x for x in STEPS if x in steps)}; "
                f"range {'all' if all_data else f'{t0} to {t1}'}", fg=typer.colors.BLUE)
    lock = logos_lock.acquire(f"ie3-{site}", log=lambda m: typer.secho(m, fg=typer.colors.YELLOW))   # held until exit

    if "import" in steps:
        # 1. Import chromatograms from incoming tarballs into GCwerks
        _run([str(bin_dir / "ie3_import.py"), "--past-days", str(past_days), site])

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
        _run([str(bin_dir / "ie3_cylinders2db.py"), "--site", site], warn=True)

    if "update" in steps:
        # Re-integrate in GCwerks from the range's month (after integration edits)
        gc_dir = f"/hats/gc/{site}"
        _run([f"{GCWERKS}/gcupdate", "-gcdir", gc_dir, ingest_common.gcupdate_yymm(gc_dir, t0, all_data)])
        _run([f"{GCWERKS}/gccalc", "-gcdir", gc_dir])

    if "load" in steps:
        # 3. Export from GCwerks and load into database. The --flagged path writes
        # GCwerks flags to hats.ng_insitu_mole_fraction_tags.
        _run([str(bin_dir / "ie3_gcwerks2db.py"), site, "--flagged", *range_args])

    if "eng" in steps:
        # 4. Upsert sample loop temp/pressure/flow from engineering data
        _run([str(bin_dir / "ie3_eng2db.py"), "--site", site, *range_args])

    if "methods" in steps:
        # 5. Set the calibration method (cal12/cal1) on any newly-ingested rows
        # before computing fits, so update_fits() builds the right kind of fit.
        _run([str(bin_dir / "ie3_set_mf_method.py")])

        # 5b. Response corrections for the AC-cycle temperature effect (temporary
        # SMO location, 2026-05-11 on) for rows that don't have one yet, so
        # ie3_batch.py below applies them. Warn only: missing factors mean
        # uncorrected (factor 1), not wrong.
        _run([str(bin_dir / "ie3_resp_correction.py"), "--new-only", "-i"], warn=True)

    if "batch" in steps:
        # 6. Recalculate and write mole fractions for all analytes
        _run([str(bin_dir / "ie3_batch.py"), "--site", site, "-p", "all", "-i", "--fits", *range_args])

    typer.secho("Ingest complete.", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
