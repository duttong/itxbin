#!/usr/bin/env python3
"""
Export M-system (M1/M3/M4) flask pair-average mole fraction data to GML format.

Without --site: queries all background sites, writes {analyte}_GCMS_flasks.txt
With --site:    single site, writes {analyte}_GCMS_flask_{site}.txt

Usage examples:
    python3 mstar-export.py --parameter COS
    python3 mstar-export.py --parameter COS --site alt
    python3 mstar-export.py --parameter COS --site mlo --start-date 2010-01-01
    python3 mstar-export.py --parameter SF6  --site brw --output /tmp/SF6_brw.txt
"""

from __future__ import annotations

import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'itxbin'))
from hats_db import HATSdb

app = typer.Typer(add_completion=False)

INSTRUMENTS = ('M1', 'M3', 'M4')
# inst_num per ccgg.inst_description (same numbers used by hats.data_exclusions
# and ng_pair_avg_view).
EXCLUSION_INST_NUM = {'M1': 46, 'M3': 54, 'M4': 192}
LOGOS_SITES = (
    'alt', 'brw', 'cgo', 'hfm', 'kum', 'lef', 'mhd',
    'mlo', 'nwr', 'psa', 'rpb', 'smo', 'spo', 'sum', 'thd',
)
MISSING = -99          # fill value for missing met data
MF_DECIMALS = 2        # mole fraction decimal places
SD_DECIMALS = 2
HEADER_FILE = Path(__file__).parent / 'mstar_header.txt'


# ── helpers ───────────────────────────────────────────────────────────────────

def decimal_year(dt: datetime) -> float:
    """Convert datetime to decimal year."""
    year = dt.year
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return year + (dt - start).total_seconds() / (end - start).total_seconds()


def fmt_float(val, max_decimals: int) -> str:
    """Format float, stripping trailing zeros. Returns MISSING sentinel for null."""
    if val is None or (isinstance(val, float) and val != val):
        return str(MISSING)
    s = f'{val:.{max_decimals}f}'.rstrip('0').rstrip('.')
    return s


def build_header(filename: str) -> str:
    """Read mstar_header.txt and fill in {filename} and {date} placeholders."""
    template = HEADER_FILE.read_text()
    return (template
            .replace('{filename}', filename)
            .replace('{date}', datetime.now().strftime('%Y-%m-%d')))


def query_exclusions(db: HATSdb, parameter_num: int) -> dict[str, list[tuple[date, date]]]:
    """Return {inst_id: [(a_start_date, a_end_date), ...]} from hats.data_exclusions.
    Half-open ranges: a row is excluded when a_start_date <= analysis_datetime < a_end_date.
    sample_type is ignored — exclusions apply to all flask data."""
    inst_nums = ', '.join(str(n) for n in EXCLUSION_INST_NUM.values())
    rows = db.doquery(
        'SELECT inst_num, a_start_date, a_end_date FROM hats.data_exclusions '
        f'WHERE parameter_num = {parameter_num} AND inst_num IN ({inst_nums})'
    )
    inv = {v: k for k, v in EXCLUSION_INST_NUM.items()}
    out: dict[str, list[tuple[date, date]]] = {}
    for r in rows or ():
        out.setdefault(inv[r['inst_num']], []).append((r['a_start_date'], r['a_end_date']))
    return out


def is_excluded(inst_id: str, dt: datetime, exclusions: dict[str, list[tuple[date, date]]]) -> bool:
    d = dt.date() if isinstance(dt, datetime) else pd.Timestamp(dt).date()
    for start, end in exclusions.get(inst_id, ()):
        if start <= d < end:
            return True
    return False


def lookup_display_name(db: HATSdb, parameter_num: int) -> str | None:
    """Return display_name from analyte_list for M4 (inst_num=192); fall back
    to any matching row if M4 not found."""
    rows = db.doquery(
        'SELECT display_name FROM hats.analyte_list '
        f'WHERE param_num = {parameter_num} AND inst_num = 192 LIMIT 1'
    )
    if rows:
        return rows[0]['display_name']
    rows = db.doquery(
        'SELECT display_name FROM hats.analyte_list '
        f'WHERE param_num = {parameter_num} LIMIT 1'
    )
    return rows[0]['display_name'] if rows else None


# ── main ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    parameter: str = typer.Option(..., help='Parameter/compound name as in ng_pair_avg_view (e.g. COS, SF6)'),
    site: Optional[str] = typer.Option(None, help='Station code (e.g. alt, mlo, brw, smo); omit for all background sites'),
    start_date: Optional[str] = typer.Option(None, help='Start date YYYY-MM-DD (inclusive)'),
    end_date: Optional[str] = typer.Option(None, help='End date YYYY-MM-DD (inclusive)'),
    output: Optional[Path] = typer.Option(None, help='Output file path'),
):
    db = HATSdb()

    # ── build query ──────────────────────────────────────────────────────────
    insts = ', '.join(f"'{i}'" for i in INSTRUMENTS)
    where = (
        f"inst_id IN ({insts}) "
        f"AND parameter = '{parameter}' "
        f"AND sample_type IN ('S', 'G', 'S85', 'SA')"
    )
    if site:
        where += f" AND UPPER(site) = '{site.upper()}'"
    else:
        site_list = ', '.join(f"'{s.upper()}'" for s in LOGOS_SITES)
        where += f" AND UPPER(site) IN ({site_list})"
    if start_date:
        where += f" AND sample_datetime >= '{start_date}'"
    if end_date:
        where += f" AND sample_datetime <= '{end_date} 23:59:59'"

    query = f'SELECT * FROM hats.ng_pair_avg_view WHERE {where} ORDER BY site, sample_datetime'
    rows = db.doquery(query)

    if not rows:
        msg = f'No data found for parameter={parameter}'
        if site:
            msg += f' site={site}'
        typer.echo(msg, err=True)
        raise typer.Exit(1)

    df = pd.DataFrame(rows)

    # ── resolve display name and output filename ─────────────────────────────
    param_num = int(df['parameter_num'].iloc[0])
    display_name = lookup_display_name(db, param_num) or parameter

    # ── apply hats.data_exclusions (drops rows in excluded inst/date ranges) ─
    exclusions = query_exclusions(db, param_num)
    if exclusions:
        n_before = len(rows)
        rows = [r for r in rows
                if not is_excluded(r['inst_id'], r['analysis_datetime'], exclusions)]
        n_excluded = n_before - len(rows)
        if n_excluded:
            typer.echo(f'Excluded {n_excluded} rows via hats.data_exclusions', err=True)
        if not rows:
            typer.echo('No data remain after applying exclusions', err=True)
            raise typer.Exit(1)

    if output:
        out_path = output
    elif site:
        out_path = Path(f'{display_name}_GCMS_flask_{site.lower()}.txt')
    else:
        out_path = Path(f'{display_name}_GCMS_flasks.txt')

    # ── build header and column line ─────────────────────────────────────────
    file_header = build_header(out_path.name)
    col_mf = display_name
    col_sd = f'{display_name}_sd'
    col_line = '\t'.join(['site', 'dec_date', 'yyyymmdd hhmmss', 'wind_dir', 'wind_spd',
                          col_mf, col_sd])

    lines = [col_line]

    for row in rows:
        dt = row['sample_datetime']
        if not isinstance(dt, datetime):
            dt = pd.Timestamp(dt).to_pydatetime()

        dec = f'{decimal_year(dt):.5f}'
        dt_str = dt.strftime('%Y%m%d %H%M')
        wind_dir = fmt_float(row['Wind_Direction'], 1)
        wind_spd = fmt_float(row['Wind_Speed'], 1)

        mf = row['pair_avg']
        sd = row['pair_stdv']
        mf_str = f'{mf:.{MF_DECIMALS}f}' if mf is not None else str(MISSING)
        sd_str = f'{sd:.{SD_DECIMALS}f}' if sd is not None else str(MISSING)

        lines.append('\t'.join([
            row['site'].lower(),
            dec,
            dt_str,
            wind_dir,
            wind_spd,
            mf_str,
            sd_str,
        ]))

    text = file_header + '\n'.join(lines) + '\n'

    out_path.write_text(text)
    n = len(lines) - 1  # subtract column header line
    typer.echo(f'Wrote {n} records to {out_path}')


if __name__ == '__main__':
    app()
