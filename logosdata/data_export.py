"""
Data export utilities for LOGOS instruments.

Classes
-------
MstarDataExporter
    Export M-system (M1/M3/M4) flask pair-average mole fraction data to
    GML-format text files.  Adapted from mstar-export.py.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

# Pseudo-site names that select PFP-only data from the named base site.
# Must stay consistent with the definition in logos_timeseries.py.
_PFP_SITES = {'MLO_PFP': 'MLO', 'MKO_PFP': 'MKO'}


HEADER_FILE = Path(__file__).parent / 'mstar_header.txt'
MISSING = -99
MF_DECIMALS = 2
SD_DECIMALS = 2


class MstarDataExporter:
    """Export M-system (M1/M3/M4) flask pair-average mole fraction data to GML format.

    Parameters
    ----------
    instrument :
        Instrument instance with a ``doquery(sql, params)`` method.
    parameter :
        Compound display name as in ``analyte_list`` (e.g. ``'CFC-11'``).
    parameter_num :
        ``parameter_num`` integer for the compound.
    sites :
        List of site codes (case-insensitive).
    start_year / end_year :
        Inclusive year range to export.
    """

    INSTRUMENTS = ('M1', 'M3', 'M4')

    def __init__(
        self,
        instrument,
        parameter: str,
        parameter_num: int,
        sites: list[str],
        start_year: int,
        end_year: int,
    ):
        self.instrument = instrument
        self.parameter = parameter
        self.parameter_num = parameter_num
        self.sites = [s.upper() for s in sites]
        self.start_year = start_year
        self.end_year = end_year

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def decimal_year(dt: datetime) -> float:
        """Convert a datetime to decimal year."""
        year = dt.year
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        return year + (dt - start).total_seconds() / (end - start).total_seconds()

    @staticmethod
    def fmt_float(val, max_decimals: int) -> str:
        """Format float, stripping trailing zeros.  Returns MISSING sentinel for null."""
        if val is None or (isinstance(val, float) and val != val):
            return str(MISSING)
        s = f'{val:.{max_decimals}f}'.rstrip('0').rstrip('.')
        return s

    # ── header ────────────────────────────────────────────────────────────────

    def build_header(self, filename: str) -> str:
        """Fill {filename} and {date} placeholders in mstar_header.txt."""
        template = HEADER_FILE.read_text()
        return (template
                .replace('{filename}', filename)
                .replace('{date}', datetime.now().strftime('%Y-%m-%d')))

    # ── data query ───────────────────────────────────────────────────────────

    def query_data(self) -> pd.DataFrame:
        """Query ng_pair_avg_view for all M* instruments, returning a DataFrame.

        Regular sites use sample_type IN ('S', 'G').  PFP pseudo-sites
        (e.g. MLO_PFP) query the base site with sample_type='PFP' and
        relabel the site column so the output carries the pseudo-site name.
        """
        insts = ', '.join(f"'{i}'" for i in self.INSTRUMENTS)
        regular_sites = [s for s in self.sites if s not in _PFP_SITES]
        pfp_pseudo_sites = [s for s in self.sites if s in _PFP_SITES]

        frames = []

        if regular_sites:
            site_list = ', '.join(f"'{s}'" for s in regular_sites)
            sql = f"""
            SELECT *
            FROM hats.ng_pair_avg_view
            WHERE inst_id IN ({insts})
              AND parameter_num = %s
              AND sample_type IN ('S', 'G')
              AND UPPER(site) IN ({site_list})
              AND YEAR(sample_datetime) BETWEEN %s AND %s
            ORDER BY site, sample_datetime
            """
            rows = self.instrument.doquery(sql, [self.parameter_num, self.start_year, self.end_year])
            frames.append(pd.DataFrame(rows) if rows else pd.DataFrame())

        for pfp_site in pfp_pseudo_sites:
            base_site = _PFP_SITES[pfp_site]
            sql = f"""
            SELECT *
            FROM hats.ng_pair_avg_view
            WHERE inst_id IN ({insts})
              AND parameter_num = %s
              AND sample_type = 'PFP'
              AND UPPER(site) = '{base_site}'
              AND YEAR(sample_datetime) BETWEEN %s AND %s
            ORDER BY sample_datetime
            """
            rows = self.instrument.doquery(sql, [self.parameter_num, self.start_year, self.end_year])
            df_pfp = pd.DataFrame(rows) if rows else pd.DataFrame()
            if not df_pfp.empty:
                df_pfp['site'] = pfp_site
            frames.append(df_pfp)

        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        if not df.empty:
            df['sample_datetime'] = pd.to_datetime(df['sample_datetime'])
            df = df.sort_values(['site', 'sample_datetime'])
        return df

    # ── format ───────────────────────────────────────────────────────────────

    def format_lines(self, df: pd.DataFrame) -> list[str]:
        """Return a list of tab-separated data lines (including the column header)."""
        col_mf = self.parameter
        col_sd = f'{self.parameter}_sd'
        col_line = '\t'.join([
            'site', 'dec_date', 'yyyymmdd hhmmss', 'wind_dir', 'wind_spd',
            col_mf, col_sd,
        ])
        lines = [col_line]
        for _, row in df.iterrows():
            dt = row['sample_datetime']
            if not isinstance(dt, datetime):
                dt = pd.Timestamp(dt).to_pydatetime()

            dec = f'{self.decimal_year(dt):.5f}'
            dt_str = dt.strftime('%Y%m%d %H%M')
            wind_dir = self.fmt_float(row.get('Wind_Direction'), 1)
            wind_spd = self.fmt_float(row.get('Wind_Speed'), 1)

            mf = row.get('pair_avg')
            sd = row.get('pair_stdv')
            mf_str = f'{mf:.{MF_DECIMALS}f}' if mf is not None else str(MISSING)
            sd_str = f'{sd:.{SD_DECIMALS}f}' if sd is not None else str(MISSING)

            lines.append('\t'.join([
                str(row['site']).lower(),
                dec,
                dt_str,
                wind_dir,
                wind_spd,
                mf_str,
                sd_str,
            ]))
        return lines

    # ── export ────────────────────────────────────────────────────────────────

    def default_filename(self) -> str:
        """Return a sensible default output filename."""
        return f'{self.parameter}_GCMS_flasks.txt'

    def export(self, output_path: str | Path) -> int:
        """Query data and write to *output_path*.

        Returns the number of data records written (0 if no data found).
        """
        df = self.query_data()
        if df.empty:
            return 0
        lines = self.format_lines(df)
        header = self.build_header(Path(output_path).name)
        Path(output_path).write_text(header + '\n'.join(lines) + '\n')
        return len(lines) - 1  # subtract the column-header line

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_timeseries_widget(
        cls, widget, sites: list[str] | None = None, all_time: bool = False
    ) -> 'MstarDataExporter':
        """Construct from a ``TimeseriesWidget`` using its current UI state.

        Pass *sites* explicitly to override the widget's active-site selection
        (e.g. to export all sites regardless of which checkboxes are checked).
        Pass *all_time=True* to export all available years regardless of the
        year-range spinboxes.
        """
        analyte = widget.analyte_combo.currentText()
        pnum = widget.analytes.get(analyte)
        return cls(
            instrument=widget.instrument,
            parameter=analyte,
            parameter_num=pnum,
            sites=sites if sites is not None else widget.get_active_sites(),
            start_year=1990 if all_time else widget.start_year.value(),
            end_year=datetime.now().year if all_time else widget.end_year.value(),
        )
