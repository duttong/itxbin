#! /usr/bin/env python
"""
Export IE3 in-situ air mole fractions from the HATS DB to one CSV file.

One row per air injection (ports 3 and 7), with a mole fraction and
uncertainty column per gas.  Rejected data and gases with no mole fraction
are written as empty fields.  CFC-11, CFC-12 and halon-1211 are measured on
two ECD channels; the preferred channel per hats.ng_preferred_channel is
used (the same rule as insitu_monthly_means_batch.py).

(ie3_export.py is the GCwerks exporter used by ie3_gcwerks2db.py.)

Usage:
    ie3_data_export.py --site smo
    ie3_data_export.py --site smo --start 2026-06-01 --end 2026-09-30 -o smo.csv
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

sys.path.append('/ccg/src/db/')
import db_utils.db_conn as db_conn  # noqa: E402

app = typer.Typer(add_completion=False)

INST_NUM = 236
AIR_PORTS = {3: 'air1', 7: 'air2'}
# IE3_Instrument.DATA_START_DATE: earlier rows are pre-deployment tests.
DATA_START_DATE = '2026-01-01'

# parameter_num -> (column name, long name, units, calibration scale, decimals)
GASES = {
    5:   ('N2O',     'N2O (nitrous oxide)',                        'ppb', 'NOAA-2006A', 2),
    6:   ('SF6',     'SF6 (sulfur hexafluoride)',                  'ppt', 'WMO X2014',  3),
    114: ('CFC-11',  'CCl3F (CFC-11, chlorofluorocarbon-11)',      'ppt', 'NOAA-2016',  2),
    22:  ('CFC-12',  'CCl2F2 (CFC-12, chlorofluorocarbon-12)',     'ppt', 'NOAA-2008',  2),
    32:  ('CFC-113', 'CCl2FCClF2 (CFC-113, chlorofluorocarbon-113)', 'ppt', 'NOAA-2002', 2),
    26:  ('H-1211',  'CBrClF2 (halon-1211)',                       'ppt', 'NOAA-2006',  3),
    37:  ('CCl4',    'CCl4 (carbon tetrachloride)',                'ppt', 'NOAA-2008',  2),
}

HEADER = """\
# File: {filename}
# Date: {date}
#
# -------------------------------------------------------------------------
# NOAA Global Monitoring Laboratory (GML), Boulder, Colorado, USA
# Long-lived Greenhouse Gases and Ozone-Depleting Substances (LOGOS) group
# Formerly known as the Halocarbons and other Atmospheric Trace Species
# (HATS) group.  https://gml.noaa.gov/hats/
#
# In-situ measurements of trace gases in ambient air by electron capture
# gas chromatography (GC-ECD).
# Instrument: IE3 (three-channel in-situ GC-ECD)
#
# Sampling site: {site_code} ({site_name} -- {lat_str}, {elev} m asl)
#   (https://gml.noaa.gov/dv/site)
#
# Data period: {first} to {last} (UTC)
#
# Contact:
#   Geoff Dutton        NOAA/GML  (720) 466-3264  geoff.dutton@noaa.gov
#
# -------------------------------------------------------------------------
# USE OF DATA
#
# For the NOAA/GML/LOGOS data use policy see:
#   https://gml.noaa.gov/hats/hats_datause.html
#
# These data are preliminary and are shared for comparison purposes.  Please
# contact the investigators before using them in a presentation,
# publication, or report, and credit NOAA/GML, Boulder, CO as the source.
#
# -------------------------------------------------------------------------
# TRACE GASES
#
# Gas       Name                                          Units  Scale       Channel
{gas_table}
#
# Calibration scales: https://gml.noaa.gov/ccl/scales.html
# Units: ppt = pmol/mol, ppb = nmol/mol (dry-air mole fraction).
#
# -------------------------------------------------------------------------
# COLUMN DESCRIPTIONS
#
# site         Station code
# datetime     Injection date and time, yyyy-mm-dd hh:mm:ss (UTC)
# dec_date     Injection date and time as decimal year (UTC)
# inlet        Air inlet: air1 (port 3) or air2 (port 7)
# <gas>        Mole fraction of a single air injection
# <gas>_unc    Estimated 1-sigma uncertainty of that value, from the
#              calibration fit and the reference-tank precision
#
# Each air injection is bracketed by reference-tank injections, and the
# response is normalized to the smoothed reference response.  Values
# rejected by quality control (instrument problems, chromatography,
# contamination, calibration tank transitions) are left empty, as are gases
# with no result for an injection.  Rows with no valid value for any gas
# are omitted.
#
# -------------------------------------------------------------------------
"""


def decimal_year(dt: pd.Series) -> pd.Series:
    """Convert a datetime Series to decimal year."""
    start = pd.to_datetime(dt.dt.year.astype(str) + '-01-01')
    end = pd.to_datetime((dt.dt.year + 1).astype(str) + '-01-01')
    return dt.dt.year + (dt - start) / (end - start)


def query_data(db, site: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Non-rejected air-port mole fractions on each gas's preferred channel."""
    pnums = ','.join(str(p) for p in GASES)
    ports = ','.join(str(p) for p in AIR_PORTS)
    end_clause = 'AND v.analysis_time < DATE_ADD(%s, INTERVAL 1 DAY)' if end else ''
    sql = f"""
    SELECT v.analysis_num, v.analysis_time, v.port, v.parameter_num,
           v.channel, v.mole_fraction, v.unc
    FROM hats.ng_insitu_data_view v
    WHERE v.inst_num = %s
      AND v.site = %s
      AND v.port IN ({ports})
      AND v.parameter_num IN ({pnums})
      AND v.rejected = 0
      AND v.mole_fraction IS NOT NULL
      AND v.analysis_time >= %s
      {end_clause}
      AND v.channel = COALESCE(
            (SELECT pc.channel FROM hats.ng_preferred_channel pc
             WHERE pc.inst_num = v.inst_num AND pc.parameter_num = v.parameter_num
               AND pc.start_date <= v.analysis_time
             ORDER BY pc.start_date DESC LIMIT 1),
            (SELECT pc.channel FROM hats.ng_preferred_channel pc
             WHERE pc.inst_num = v.inst_num AND pc.parameter_num = v.parameter_num
             ORDER BY pc.start_date ASC LIMIT 1),
            v.channel)
    ORDER BY v.analysis_time
    """
    params = [INST_NUM, site, start] + ([end] if end else [])
    df = pd.DataFrame(db.doquery(sql, params))
    if df.empty:
        return df
    df['analysis_time'] = pd.to_datetime(df['analysis_time'])
    for col in ('mole_fraction', 'unc'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def to_wide(df: pd.DataFrame, site: str) -> pd.DataFrame:
    """One row per injection, <gas> and <gas>_unc columns in GASES order."""
    idx = ['analysis_num', 'analysis_time', 'port']
    wide = df.pivot_table(index=idx, columns='parameter_num',
                          values=['mole_fraction', 'unc'], aggfunc='first')
    out = wide.index.to_frame(index=False)
    out.insert(0, 'site', site.upper())
    out['datetime'] = out['analysis_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    out['dec_date'] = decimal_year(out['analysis_time']).round(6)
    out['inlet'] = out['port'].map(AIR_PORTS)
    for pnum, (name, _, _, _, dec) in GASES.items():
        for value, suffix in (('mole_fraction', ''), ('unc', '_unc')):
            col = (value, pnum)
            vals = wide[col].to_numpy() if col in wide.columns else float('nan')
            out[name + suffix] = pd.Series(vals, index=out.index).round(dec)
    cols = ['site', 'datetime', 'dec_date', 'inlet'] + [
        name + suffix for name, *_ in GASES.values() for suffix in ('', '_unc')]
    return out.sort_values('analysis_time')[cols]


def channel_text(df: pd.DataFrame, pnum: int) -> str:
    """Channels used for a gas, with date ranges when it changed."""
    g = df[df['parameter_num'] == pnum].sort_values('analysis_time')
    if g.empty:
        return '-'
    runs = g['channel'].ne(g['channel'].shift()).cumsum()
    spans = g.groupby(runs).agg(ch=('channel', 'first'),
                                t0=('analysis_time', 'min'), t1=('analysis_time', 'max'))
    if len(spans) == 1:
        return spans['ch'].iat[0]
    return ', '.join(f"{r.ch} ({r.t0:%Y-%m-%d} to {r.t1:%Y-%m-%d})" for r in spans.itertuples())


def build_header(db, filename: str, site: str, df: pd.DataFrame) -> str:
    rows = db.doquery('SELECT name, lat, lon, elev FROM gmd.site WHERE code = %s LIMIT 1',
                      [site.upper()])
    info = rows[0] if rows else {}
    lat = float(info.get('lat', float('nan')))
    lon = float(info.get('lon', float('nan')))
    lat_str = f"{abs(lat):.2f} {'N' if lat >= 0 else 'S'}, {abs(lon):.2f} {'E' if lon >= 0 else 'W'}"
    gas_table = '\n'.join(
        f'# {name:<9} {long:<45} {units:<6} {scale:<11} {channel_text(df, pnum)}'
        for pnum, (name, long, units, scale, _) in GASES.items())
    return HEADER.format(
        filename=filename,
        date=datetime.now().strftime('%Y-%m-%d'),
        site_code=site.upper(),
        site_name=info.get('name', site.upper()),
        lat_str=lat_str,
        elev=f"{float(info.get('elev', float('nan'))):.0f}",
        first=f"{df['analysis_time'].min():%Y-%m-%d}",
        last=f"{df['analysis_time'].max():%Y-%m-%d}",
        gas_table=gas_table,
    )


@app.command()
def main(
    site: str = typer.Option('smo', '--site', help='Station code of the IE3 instrument.'),
    start: str = typer.Option(DATA_START_DATE, '--start', help='First date, YYYY-MM-DD.'),
    end: Optional[str] = typer.Option(None, '--end', help='Last date (inclusive), YYYY-MM-DD.'),
    output: Optional[Path] = typer.Option(None, '--output', '-o',
                                          help='Output CSV (default IE3_<SITE>_insitu.csv).'),
):
    site = site.lower()
    db = db_conn.HATS_ng()
    df = query_data(db, site, start, end)
    if df.empty:
        typer.secho(f'No IE3 data for site {site}', fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    out_path = output or Path(f'IE3_{site.upper()}_insitu.csv')
    wide = to_wide(df, site)
    with open(out_path, 'w') as f:
        f.write(build_header(db, out_path.name, site, df))
        wide.to_csv(f, index=False, na_rep='')
    typer.echo(f'Wrote {len(wide)} injections to {out_path}')


if __name__ == '__main__':
    app()
