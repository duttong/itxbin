#!/usr/bin/env python3
"""
Load the IE3 operator cylinder-pressure log into hats.ng_cylinder_pressures.

The log is written by the Cylinders tab in IE3_acquisition/display.py
(columns Date, Cylinder, Pressure; Date is UTC, 'YYYY-MM-DD HH:MM') and copied
to /hats/gc/{site}/logs/ by ie3_ingest.py.

- Cal tank serials are respelled to match reftank.fill (operators sometimes
  type "ALM-064967" as "ALM064967" or vice versa).
- 'N2' and 'CO2' rows are utility cylinders (carrier / dopant); see
  hats.ng_cylinder_types.
- serial_number is the physical tank serial: the same as cylinder for cal
  tanks; left NULL for utility cylinders (the log doesn't record their serials).
- Rows are upserted on (inst_num, cylinder, reading_datetime).  When the log has
  the same cylinder twice in one minute the last line wins, matching an operator
  re-save.  Only `pressure` (and a blank serial_number) is updated on conflict,
  so `rejected`, `comment` and serials set in the DB are never overwritten.
- Raw readings are stored as logged; the "re-entered within 60 minutes"
  correction rule is applied by readers (dashboard / display.py), not here.

Usage:
    python3 ie3_cylinders2db.py [--site smo] [--file PATH] [--dry-run]
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import typer

INST_NUM_BY_SITE = {'smo': 236}
UTILITY_TYPES = {'N2': 2, 'CO2': 3}     # ng_cylinder_types.num; everything else is cal (1)
CAL_TYPE = 1

app = typer.Typer(add_completion=False)


def serial_key(serial: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', serial).upper()


def read_log(path: Path) -> tuple[dict, int]:
    """Return ({(cylinder, datetime): pressure}, n_skipped); later lines win."""
    readings = {}
    skipped = 0
    with path.open(newline='') as fh:
        for row in csv.DictReader(fh):
            name = (row.get('Cylinder') or '').strip()
            try:
                when = datetime.strptime((row.get('Date') or '').strip(), '%Y-%m-%d %H:%M')
                psi = float((row.get('Pressure') or '').strip())
            except ValueError:
                if name:
                    skipped += 1
                    print(f'Skipping unparseable row: {row}', file=sys.stderr)
                continue
            if name:
                readings[(name, when)] = psi
    return readings, skipped


@app.command()
def main(
    site: str = typer.Option('smo', '--site', help='Station code for the IE3 instrument.'),
    file: Path = typer.Option(None, '--file', help='Log to load (default /hats/gc/{site}/logs/cylinder_pressures.csv).'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Parse and report without writing.'),
):
    """Upsert IE3 cylinder pressure readings into hats.ng_cylinder_pressures."""
    site = site.lower()
    if site not in INST_NUM_BY_SITE:
        raise typer.BadParameter(f'No inst_num for site {site!r}; add it to INST_NUM_BY_SITE.')
    inst_num = INST_NUM_BY_SITE[site]
    path = file or Path(f'/hats/gc/{site}/logs/cylinder_pressures.csv')
    if not path.exists():
        typer.secho(f'No cylinder log at {path}; nothing to load.', fg=typer.colors.YELLOW, err=True)
        raise typer.Exit()

    sys.path.append('/ccg/src/db/')
    import db_utils.db_conn as db_conn  # type: ignore
    db = db_conn.HATS_ng()
    site_num = db.doquery('SELECT num FROM gmd.site WHERE code = %s', [site.upper()])[0]['num']
    fill_serials = {serial_key(r['serial_number']): r['serial_number']
                    for r in db.doquery('SELECT DISTINCT serial_number FROM reftank.fill')}

    readings, skipped = read_log(path)
    params = []
    unknown = set()
    for (name, when), psi in sorted(readings.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        type_num = UTILITY_TYPES.get(name.upper(), CAL_TYPE)
        cylinder = name.upper() if type_num != CAL_TYPE else fill_serials.get(serial_key(name))
        if cylinder is None:
            unknown.add(name)
            cylinder = name
        serial = cylinder if type_num == CAL_TYPE else None
        params.append((inst_num, site_num, type_num, cylinder, serial, when.strftime('%Y-%m-%d %H:%M:00'), psi))

    print(f'{path}: {len(params)} readings, {skipped} unparseable row(s) skipped')
    if unknown:
        print(f'Warning: not in reftank.fill (loaded as typed): {", ".join(sorted(unknown))}', file=sys.stderr)
    if dry_run or not params:
        return

    sql = """
        INSERT INTO hats.ng_cylinder_pressures
            (inst_num, site_num, cylinder_type_num, cylinder, serial_number, reading_datetime, pressure)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE pressure = VALUES(pressure),
                                serial_number = COALESCE(serial_number, VALUES(serial_number))
    """
    db.doMultiInsert(sql, params, all=True)
    n = db.doquery('SELECT COUNT(*) AS n FROM hats.ng_cylinder_pressures WHERE inst_num = %s', [inst_num])[0]['n']
    print(f'Upserted into hats.ng_cylinder_pressures; {n} rows now for inst_num {inst_num}')


if __name__ == '__main__':
    app()
