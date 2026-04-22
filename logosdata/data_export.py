"""
Data export utilities for LOGOS instruments.

Classes
-------
MstarDataExporter
    Export M-system (M1/M3/M4) flask pair-average mole fraction data to
    GML-format text files.  Adapted from mstar-export.py.
FecdDataExporter
    Export fECD (OTTO + FE3) flask pair-average mole fraction data to
    GML-format text files, one file per site.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()
    if len(non_empty) == 1:
        return non_empty[0].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return pd.concat(non_empty, ignore_index=True)

# Pseudo-site names that select PFP-only data from the named base site.
# Must stay consistent with the definition in logos_timeseries.py.
_PFP_SITES = {'MLO_PFP': 'MLO', 'MKO_PFP': 'MKO'}


HEADER_FILE = Path(__file__).parent / 'mstar_header.txt'
MISSING = 'nd'
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

        Regular sites use sample_type IN ('S', 'G', 'S85', 'SA').  PFP pseudo-sites
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
              AND sample_type IN ('S', 'G', 'S85', 'SA')
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

        df = _concat_frames(frames)
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


FECD_HEADER_FILE = Path(__file__).parent / 'fecd_header.txt'

# Long chemical names for header text, keyed by parameter name from the view.
_FECD_CHEM_NAMES = {
    'CFC11':   'CCl3F (CFC-11, Chlorofluorocarbon-11)',
    'CFC12':   'CCl2F2 (CFC-12, Chlorofluorocarbon-12)',
    'CFC113':  'CCl2FCClF2 (CFC-113, Chlorofluorocarbon-113)',
    'CCl4':    'CCl4 (Carbon Tetrachloride)',
    'CH3CCl3': 'CH3CCl3 (Methyl Chloroform, 1,1,1-Trichloroethane)',
    'CHCl3':   'CHCl3 (Chloroform)',
    'H1211':   'CBrClF2 (Halon-1211)',
    'n2o':     'N2O (Nitrous Oxide)',
    'sf6':     'SF6 (Sulfur Hexafluoride)',
    'TCE':     'C2HCl3 (Trichloroethylene)',
}

# Short codes used in filenames and column headers (e.g. "F11" for CFC11).
_FECD_SHORT_CODES = {
    'CFC11':   'F11',
    'CFC12':   'F12',
    'CFC113':  'F113',
    'CCl4':    'CCl4',
    'CH3CCl3': 'MC',
    'CHCl3':   'CHCl3',
    'H1211':   'H1211',
    'n2o':     'N2O',
    'sf6':     'SF6',
    'TCE':     'TCE',
}


class FecdDataExporter:
    """Export fECD (OTTO + FE3) flask pair-average data to GML-format text files.

    Writes one file per site named {site}_{analyte}_All.txt.  The analyte
    name in the filename is the bare parameter name without a channel suffix.

    For analytes with a preferred channel in ng_preferred_channel (e.g.
    CFC11, CFC113), FE3 data is filtered to that channel from start_date
    onward (no channel filter before start_date).  OTTO never has duplicate
    channels so it is always queried without a channel filter.

    Parameters
    ----------
    instrument :
        FE3_Instrument instance (inst_num=193).
    parameter_num :
        parameter_num integer for the compound.
    parameter_name :
        Bare parameter name (no channel suffix), e.g. ``'CFC11'``.
    sites :
        List of site codes (case-insensitive).
    start_year / end_year :
        Inclusive year range to export.
    """

    FE3_INST_NUM = 193

    def __init__(
        self,
        instrument,
        parameter_num: int,
        parameter_name: str,
        sites: list[str],
        start_year: int,
        end_year: int,
    ):
        self.instrument = instrument
        self.parameter_num = parameter_num
        self.parameter_name = parameter_name  # bare name, no channel
        self.sites = [s.upper() for s in sites]
        self.start_year = start_year
        self.end_year = end_year
        self._preferred: Optional[dict] = None  # loaded on first use

    # ── preferred channel ────────────────────────────────────────────────────

    def _load_preferred_channel(self) -> Optional[dict]:
        """Return preferred-channel row for this parameter_num on FE3, or None."""
        rows = self.instrument.doquery(
            'SELECT channel, start_date FROM hats.ng_preferred_channel '
            'WHERE inst_num = %s AND parameter_num = %s LIMIT 1',
            [self.FE3_INST_NUM, self.parameter_num]
        )
        if rows:
            return rows[0]
        return None

    @property
    def preferred(self) -> Optional[dict]:
        if self._preferred is None:
            self._preferred = self._load_preferred_channel() or {}
        return self._preferred or None

    # ── site metadata ────────────────────────────────────────────────────────

    def _site_info(self, site: str) -> dict:
        """Return lat, lon, elev, name for a site code (empty dict if not found)."""
        rows = self.instrument.doquery(
            'SELECT name, lat, lon, elev FROM gmd.site WHERE code = %s LIMIT 1',
            [site.upper()]
        )
        return rows[0] if rows else {}

    # ── data query ───────────────────────────────────────────────────────────

    def query_site_data(self, site: str) -> pd.DataFrame:
        """Query OTTO + FE3 unflagged pair averages for one site, applying the
        preferred-channel rule for FE3 where applicable."""
        site_upper = site.upper()
        site_list_sql = '%s'

        frames = []

        # ── OTTO: all years, no channel filter ───────────────────────────────
        sql = """
        SELECT sample_datetime, pair_avg, pair_stdv, pair_id_num, sample_type,
               'OTTO' AS instrument
        FROM hats.ng_pair_avg_view
        WHERE inst_id = 'OTTO'
          AND parameter_num = %s
          AND sample_type IN ('S', 'G', 'S85', 'SA')
          AND UPPER(site) = %s
          AND YEAR(sample_datetime) BETWEEN %s AND %s
        ORDER BY sample_datetime
        """
        rows = self.instrument.doquery(sql, [self.parameter_num, site_upper,
                                             self.start_year, self.end_year])
        frames.append(pd.DataFrame(rows) if rows else pd.DataFrame())

        # ── FE3: apply preferred-channel rule ────────────────────────────────
        pref = self.preferred
        if pref:
            channel = pref['channel']
            pref_start = pref['start_date']  # date object from DB

            # Before preferred channel start_date: no channel filter
            sql_before = """
            SELECT sample_datetime, pair_avg, pair_stdv, pair_id_num, sample_type,
                   'FE3' AS instrument
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
              AND sample_type IN ('S', 'G', 'S85', 'SA')
              AND UPPER(site) = %s
              AND YEAR(sample_datetime) BETWEEN %s AND %s
              AND sample_datetime < %s
            ORDER BY sample_datetime
            """
            rows = self.instrument.doquery(sql_before, [
                self.FE3_INST_NUM, self.parameter_num, site_upper,
                self.start_year, self.end_year, pref_start
            ])
            frames.append(pd.DataFrame(rows) if rows else pd.DataFrame())

            # From preferred channel start_date onward: filter to preferred channel
            sql_after = """
            SELECT sample_datetime, pair_avg, pair_stdv, pair_id_num, sample_type,
                   'FE3' AS instrument
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
              AND channel = %s
              AND sample_type IN ('S', 'G', 'S85', 'SA')
              AND UPPER(site) = %s
              AND YEAR(sample_datetime) BETWEEN %s AND %s
              AND sample_datetime >= %s
            ORDER BY sample_datetime
            """
            rows = self.instrument.doquery(sql_after, [
                self.FE3_INST_NUM, self.parameter_num, channel, site_upper,
                self.start_year, self.end_year, pref_start
            ])
            frames.append(pd.DataFrame(rows) if rows else pd.DataFrame())

        else:
            # No preferred channel: query FE3 without channel filter
            sql = """
            SELECT sample_datetime, pair_avg, pair_stdv, pair_id_num, sample_type,
                   'FE3' AS instrument
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
              AND sample_type IN ('S', 'G', 'S85', 'SA')
              AND UPPER(site) = %s
              AND YEAR(sample_datetime) BETWEEN %s AND %s
            ORDER BY sample_datetime
            """
            rows = self.instrument.doquery(sql, [
                self.FE3_INST_NUM, self.parameter_num, site_upper,
                self.start_year, self.end_year
            ])
            frames.append(pd.DataFrame(rows) if rows else pd.DataFrame())

        df = _concat_frames(frames)
        if df.empty:
            return df
        df['sample_datetime'] = pd.to_datetime(df['sample_datetime'])
        df = df.sort_values('sample_datetime').reset_index(drop=True)
        return df

    # ── header ────────────────────────────────────────────────────────────────

    def build_header(self, filename: str, site: str) -> str:
        """Build the file header for a given site."""
        template = FECD_HEADER_FILE.read_text()
        info = self._site_info(site)
        name = info.get('name', site.upper())
        lat  = info.get('lat', float('nan'))
        lon  = info.get('lon', float('nan'))
        elev = info.get('elev', float('nan'))
        lat_str = f"{abs(lat):.2f} {'N' if lat >= 0 else 'S'}, {abs(lon):.2f} {'E' if lon >= 0 else 'W'}"
        chem_name = _FECD_CHEM_NAMES.get(self.parameter_name, self.parameter_name)
        return (template
                .replace('{filename}', filename)
                .replace('{parameter_name}', chem_name)
                .replace('{site_code}', site.upper())
                .replace('{site_name}', name)
                .replace('{lat_str}', lat_str)
                .replace('{elev}', f'{elev:.0f}')
                .replace('{date}', datetime.now().strftime('%Y-%m-%d')))

    def _short_code(self) -> str:
        return _FECD_SHORT_CODES.get(self.parameter_name, self.parameter_name)

    # ── format ───────────────────────────────────────────────────────────────

    def format_lines(self, df: pd.DataFrame, site: str) -> list[str]:
        """Return fixed-width data lines plus a column-name header line."""
        tag = f'{self._short_code()}fecd{site.upper()}'
        col_header = (
            f'{tag}yr {tag}mon {tag}day {tag}hour {tag}min '
            f'{tag}m {tag}sd {tag}pid {tag}ftype {tag}inst'
        )
        lines = [col_header]
        for _, row in df.iterrows():
            dt = row['sample_datetime']
            if not isinstance(dt, datetime):
                dt = pd.Timestamp(dt).to_pydatetime()
            mf = row['pair_avg']
            sd = row['pair_stdv']
            mf_str = f'{mf:9.3f}' if mf is not None and mf == mf else '      nan'
            sd_str = f'{sd:8.3f}' if sd is not None and sd == sd else '     nan'
            pid = row.get('pair_id_num', '')
            ftype = str(row.get('sample_type', '')).strip()
            inst = str(row.get('instrument', '')).strip()
            lines.append(
                f'{dt.year:>4}  {dt.month:>2}  {dt.day:>2}  {dt.hour:>4}  {dt.minute:>3}  '
                f'{mf_str}  {sd_str}  {int(pid):>6}  {ftype}  {inst}'
            )
        return lines

    # ── export ────────────────────────────────────────────────────────────────

    def default_filename(self, site: str) -> str:
        return f'{self._short_code()}_{site.upper()}_NOAAflaskECD_All.txt'

    def export_site(self, site: str, output_path: str | Path) -> int:
        """Export data for one site to output_path. Returns records written."""
        df = self.query_site_data(site)
        if df.empty:
            return 0
        filename = Path(output_path).name
        header = self.build_header(filename, site)
        lines = self.format_lines(df, site)
        Path(output_path).write_text(header + '\n'.join(lines) + '\n')
        return len(lines) - 1  # exclude column header line

    def export_all(self, output_dir: str | Path) -> dict[str, int]:
        """Export each site to its own file in output_dir.

        Returns a dict mapping site code to number of records written
        (sites with no data are omitted).
        """
        output_dir = Path(output_dir)
        results = {}
        for site in self.sites:
            path = output_dir / self.default_filename(site)
            n = self.export_site(site, path)
            if n > 0:
                results[site] = n
        return results

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_timeseries_widget(
        cls,
        widget,
        sites: list[str] | None = None,
        all_time: bool = False,
    ) -> 'FecdDataExporter':
        """Construct from a TimeseriesWidget (FE3 only)."""
        analyte = widget.analyte_combo.currentText()
        # Strip channel suffix to get the bare parameter name, e.g. "CFC11 (c)" -> "CFC11"
        param_name = analyte.split('(')[0].strip() if '(' in analyte else analyte.strip()
        pnum = widget.analytes.get(analyte)
        return cls(
            instrument=widget.instrument,
            parameter_num=pnum,
            parameter_name=param_name,
            sites=sites if sites is not None else widget.get_active_sites(),
            start_year=1990 if all_time else widget.start_year.value(),
            end_year=datetime.now().year if all_time else widget.end_year.value(),
        )
