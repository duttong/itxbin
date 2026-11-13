#!/usr/bin/env python3
"""Ingest CATS GC data: export from GCwerks (or logosint), load into DB, recompute mole
fractions, optionally import legacy published mole fractions.

Which integration is used comes from hats.ng_integ_source (integ_source.py), set from the
logos_integ page's Admin menu: no row = GCwerks. For a logosint site it exports with
logosint_export.py (the site's logosint parameter set, to /hats/gc/cats_results/logosint/) and
loads with cats_gcwerks2db.py --source logosint (no GCwerks flags; tag 324 left as it is), never
before the site's logosint_from date; channels logosint doesn't integrate (cc) are still loaded
from GCwerks' export. --all on a logosint site means from logosint_from."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

import integ_source
import logosint_lock
from cats_gcwerks2db import CATS_GCwerks2DB

app = typer.Typer(add_completion=False, help="Ingest CATS GC data into the HATS database.")

BIN_DIR = Path(__file__).resolve().parent
CATS_EXPORT = Path("/home/hats/gdutton/bin/cats_export.py")
LOGOSINT_EXPORT = BIN_DIR / "logosint_export.py"

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
    skip_export: bool = typer.Option(False, "--skip-export", help="Skip the CSV export step"),
    source: str | None = typer.Option(None, "--source", hidden=True,
                                      help="no longer used: the source comes from hats.ng_integ_source; if given, it must agree"),
    run_aftp: bool = typer.Option(
        False, "--run-aftp", help="Also import legacy published mole fractions from /aftp (one-time backfill; off by default)"
    ),
    mf: bool = typer.Option(
        True, "--mf/--no-mf",
        help="Recompute weekly cal fits + mole fractions via cats_batch.py (default: on)",
    ),
    set_methods: bool = typer.Option(
        False, "--set-methods",
        help="Before recomputing, reset every air row in the window to its "
             "analyte's default cal method (cats_set_mf_method.py). Off by "
             "default: it overwrites methods set by hand. New rows already "
             "get the analyte's current method from cats_gcwerks2db.py.",
    ),
):
    """Ingest CATS data for one site.

    Steps:
      1. Export GCwerks data to /hats/gc/cats_results/ (cats_export.py), or with --source
         logosint, logosint's to /hats/gc/cats_results/logosint/ (logosint_export.py)
      2. Load the CSVs into ng_insitu_analysis / ng_insitu_mole_fractions
      3. (optional, --run-aftp) Import legacy published mole fractions from /aftp/hats
      4. (default, --mf) Recompute weekly cal fits and mole fractions
         (cats_batch.py); with --set-methods, first reset the window's cal
         methods to the defaults (cats_set_mf_method.py)
    """
    inst_num = CATS_GCwerks2DB.INST_NUM_BY_SITE.get(site)
    if inst_num is None:
        typer.secho(f"unknown CATS site {site!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        src = integ_source.current(inst_num, strict=True)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if source is not None and source != src.source:
        typer.secho(f"--source {source} disagrees with hats.ng_integ_source ({site}: {src.source}); "
                    "change it on the logos_integ page's Admin menu, and drop --source", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    logosint = src.logosint
    frm = src.logosint_from
    typer.secho(f"{site.upper()}: {src.source}" + (f" from {frm:%Y-%m-%d}" if logosint and frm is not None else ""),
                fg=typer.colors.BLUE)
    if logosint:
        # One logosint run per site at a time: the reintegration worker (logosint_jobs.py) holds the
        # same lock for a whole job. Held until this process exits.
        _site_lock = logosint_lock.acquire(site, log=lambda m: typer.secho(m, fg=typer.colors.YELLOW))

    # 1. Export (GCwerks, or logosint for the same window)
    # A logosint site's load window, never before logosint_from (None = the loader's default).
    l_start = l_end = None
    if logosint:
        if all_data:
            l_start = frm
        elif year is not None:
            l_start, l_end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
            if frm is not None and l_end <= frm:
                typer.secho(f"{year} is before {site}'s logosint_from ({frm:%Y-%m-%d}): nothing to do",
                            fg=typer.colors.YELLOW)
                raise typer.Exit(code=0)
        if frm is not None and l_start is not None and l_start < frm:
            l_start = frm
    if not skip_export:
        if logosint:
            cmd = [sys.executable, str(LOGOSINT_EXPORT), site]
            if l_start is not None:
                cmd += ["--start", f"{l_start:%y%m}"]
            if l_end is not None:
                cmd += ["--end", f"{l_end:%y%m}"]
        else:
            cmd = [sys.executable, str(CATS_EXPORT), site]
            if flagged:
                cmd.append("--flagged")
        _run(cmd)

    # 2. Load results into DB
    gcwerks_cmd = [sys.executable, str(BIN_DIR / "cats_gcwerks2db.py"), site]
    if logosint:
        gcwerks_cmd += ["--source", "logosint", "--no-flagged"]
        if l_start is not None:
            gcwerks_cmd += ["--start", f"{l_start:%Y-%m-%d %H:%M}"]
        if l_end is not None:
            gcwerks_cmd += ["--end", f"{l_end:%Y-%m-%d %H:%M}"]
    else:
        if flagged:
            gcwerks_cmd.append("--flagged")
        if all_data:
            gcwerks_cmd.append("--all")
        elif year is not None:
            gcwerks_cmd.extend(["--year", str(year)])
    _run(gcwerks_cmd)
    if logosint:
        # Channels logosint doesn't integrate still come from GCwerks' export (cats_export.py, run
        # separately), with its flags, over the same window.
        cc_cmd = [sys.executable, str(BIN_DIR / "cats_gcwerks2db.py"), site, "--channel", ",".join(integ_source.GCWERKS_ONLY_CHANNELS)]
        if flagged:
            cc_cmd.append("--flagged")
        if all_data:
            cc_cmd.append("--all")
        elif year is not None:
            cc_cmd.extend(["--year", str(year)])
        _run(cc_cmd)

    # 3. Import legacy published mole fractions from /aftp/hats (one-time backfill, opt-in)
    if run_aftp:
        aftp_cmd = [sys.executable, str(BIN_DIR / "cats_aftp2db.py"), site]
        if all_data:
            aftp_cmd.append("--all")
        elif year is not None:
            aftp_cmd.extend(["--year", str(year)])
        _run(aftp_cmd)

    # 4. Recompute mole fractions: weekly fits + mole fractions. New rows
    # already carry their analyte's current cal method (set on insert by
    # cats_gcwerks2db.py). --set-methods resets the window to the defaults
    # first; it shares the window so a row is only ever tagged cal12/cal1 in
    # the same run that recomputes it.
    if mf:
        start_date, end_date = _resolve_mf_window(all_data, year)
        if logosint and all_data and frm is not None:
            start_date = f"{frm:%Y-%m-%d}"   # earlier data weren't touched

        if set_methods:
            method_cmd = [
                sys.executable, str(BIN_DIR / "cats_qc" / "cats_set_mf_method.py"),
                "--site", site, "--start", start_date,
            ]
            if end_date:
                method_cmd.extend(["--end", end_date])
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
