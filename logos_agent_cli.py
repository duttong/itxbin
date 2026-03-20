#!/usr/bin/env python3
"""
CLI entry point for read-only LOGOS data agent tools.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import typer
from typing_extensions import Annotated

# Avoid matplotlib cache warnings caused by indirect imports in logos_instruments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

from logos_agent_tools import LOGOSDataAgentTools
from logos_instruments import LOGOS_Instruments

app = typer.Typer(
    help="Read-only CLI for LOGOS agent-oriented HATS data queries.",
    context_settings={"help_option_names": ["-h", "--help"]},
)

state = {"inst_id": "fe3"}


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ")
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _print_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=_json_default, sort_keys=False))


def _tools() -> LOGOSDataAgentTools:
    return LOGOSDataAgentTools(inst_id=state["inst_id"])


@app.callback()
def main_callback(
    inst: Annotated[
        str,
        typer.Option(
            "--inst",
            "-i",
            help="Instrument ID to use for the query.",
            autocompletion=lambda: list(LOGOS_Instruments.INSTRUMENTS.keys()),
        ),
    ] = "fe3"
) -> None:
    state["inst_id"] = inst


@app.command("site-info")
def site_info(
    site_query: Annotated[str, typer.Argument(help="GML site code or site name, for example SMO or Harvard Forest.")]
) -> None:
    """Return site metadata."""
    _print_json(_tools().get_site_info(site_query))


@app.command("list-sites")
def list_sites(
    min_lat: Annotated[float | None, typer.Option("--min-lat", help="Minimum latitude, in degrees.")] = None,
    max_lat: Annotated[float | None, typer.Option("--max-lat", help="Maximum latitude, in degrees.")] = None,
    min_lon: Annotated[float | None, typer.Option("--min-lon", help="Minimum longitude, in degrees.")] = None,
    max_lon: Annotated[float | None, typer.Option("--max-lon", help="Maximum longitude, in degrees.")] = None,
    country: Annotated[str | None, typer.Option("--country", help="Country name exactly as stored in gmd.site.country.")] = None,
    continent: Annotated[str | None, typer.Option("--continent", help="Continent name, for example Africa.")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum number of sites to return.")] = 500,
) -> None:
    """Return sites with optional geographic, country, and continent filtering."""
    _print_json(
        _tools().list_sites(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            country=country,
            continent=continent,
            limit=limit,
        )
    )


@app.command("site-countries")
def site_countries() -> None:
    """Return distinct gmd.site.country values and their continent mapping."""
    _print_json(_tools().list_site_countries())


@app.command("supported-analytes")
def supported_analytes() -> None:
    """Return analytes available for the selected instrument."""
    _print_json(_tools().list_supported_analytes())


@app.command("resolve-analyte")
def resolve_analyte(
    analyte: Annotated[str, typer.Argument(help="Analyte name, for example CFC-11.")]
) -> None:
    """Resolve an analyte name to a parameter number."""
    _print_json(_tools().resolve_analyte(analyte))


@app.command("recent-pairs")
def recent_pairs(
    site_code: Annotated[str, typer.Argument(help="GML site code, for example SMO.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of recent pair runs to return.")] = 10,
) -> None:
    """Return recent flask-pair runs for a site."""
    _print_json(_tools().get_recent_flask_pairs(site_code, limit=limit))


@app.command("recent-processed-pairs")
def recent_processed_pairs(
    site_code: Annotated[str, typer.Argument(help="GML site code, for example SMO.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of recent processed pair runs to return.")] = 10,
) -> None:
    """Return recent processed/analyzed flask pairs for a site."""
    _print_json(_tools().get_recent_processed_flask_pairs(site_code, limit=limit))


@app.command("pair-metadata")
def pair_metadata(
    pair_id_num: Annotated[int, typer.Argument(help="Pair ID number from ng_data_processing_view.")]
) -> None:
    """Return Status_MetData metadata for a flask pair."""
    _print_json(_tools().get_pair_metadata(pair_id_num))


@app.command("recent-pairs-met")
def recent_pairs_met(
    site_code: Annotated[str, typer.Argument(help="GML site code, for example SMO.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of recent pair runs to return.")] = 10,
) -> None:
    """Return recent flask pairs enriched with flask type and met data."""
    _print_json(_tools().get_recent_flask_pairs_with_metadata(site_code, limit=limit))


@app.command("recent-processed-pairs-met")
def recent_processed_pairs_met(
    site_code: Annotated[str, typer.Argument(help="GML site code, for example SMO.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of recent processed pair runs to return.")] = 10,
) -> None:
    """Return recent processed/analyzed flask pairs enriched with flask type and met data."""
    _print_json(_tools().get_recent_processed_flask_pairs_with_metadata(site_code, limit=limit))


@app.command("window-mean")
def window_mean(
    site_code: Annotated[str, typer.Argument(help="GML site code, for example SMO.")],
    analyte: Annotated[str, typer.Argument(help="Analyte name, for example CFC-11.")],
    start_date: Annotated[str, typer.Argument(help="Inclusive start date, YYYY-MM-DD.")],
    end_date: Annotated[str, typer.Argument(help="Exclusive end date, YYYY-MM-DD.")],
) -> None:
    """Return flask summary statistics for a date window."""
    _print_json(_tools().get_site_flask_mean(site_code, analyte, start_date, end_date))


@app.command("recent-values")
def recent_values(
    site_code: Annotated[str, typer.Argument(help="GML site code, for example SMO.")],
    analyte: Annotated[str, typer.Argument(help="Analyte name, for example CFC-11.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of recent rows to return.")] = 10,
) -> None:
    """Return recent good flask values for a site and analyte."""
    _print_json(_tools().get_recent_flask_values(site_code, analyte, limit=limit))


@app.command("compare-year")
def compare_year(
    site_code: Annotated[str, typer.Argument(help="GML site code, for example SMO.")],
    analyte: Annotated[str, typer.Argument(help="Analyte name, for example CFC-11.")],
    year: Annotated[int, typer.Argument(help="Calendar year, for example 2025.")],
    recent_limit: Annotated[int, typer.Option("--recent-limit", "-n", help="Number of recent rows to compare against.")] = 10,
) -> None:
    """Compare a yearly site mean to the most recent flask values."""
    _print_json(_tools().compare_site_year_to_recent(site_code, analyte, year, recent_limit=recent_limit))


if __name__ == "__main__":
    app()
