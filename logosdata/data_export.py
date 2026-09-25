"""
Data export utilities for LOGOS instruments.

Classes
-------
MstarDataExporter
    Export M-system (M1/M3/M4) flask pair-average mole fraction data to
    GML-format text files.  Adapted from mstar-export.py.
MstarMonthlyExporter
    Export M-system (M1/M3/M4) monthly means of those flask pair averages,
    with a continuous month series per site.
MstarGlobalMeansExporter
    Export monthly global, hemispheric and semi-hemispheric means of the
    M-system flask pair data, with the background-site means behind them.
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

from global_means import GlobalMeansCalculator, GlobalMeansConfig


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()
    if len(non_empty) == 1:
        return non_empty[0].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return pd.concat(non_empty, ignore_index=True)

# PFP pseudo-sites, mapped to the base site they are stored under.
#
# PFP (programmable flask package) samples are a different kind of flask: they
# carry a ccgg_event_num instead of a pair_id_num.  ng_pair_avg_view admits them
# through the `ccgg_event_num > 0` branch of its WHERE clause, but files them
# under the base site, so MLO's rows are a mix of programmatic HatsFlask pairs
# and PFP pairs.  The view exposes no run_type_num, so the pseudo-site the GUI
# shows has to be rebuilt from pair_id_num instead: because the view only admits
# a row when `pair_id_num > 0 OR ccgg_event_num > 0`, `pair_id_num = 0` inside it
# necessarily means a CCGG event flask.  Only MLO and MKO carry such rows (M3 and
# M4 only), which is why the relabelling is scoped to them.
#
# Background: PFPs were deployed at MLO in 2021 and became the only sampling
# there after the Nov 2022 Mauna Loa eruption cut power and access to the
# observatory (no programmatic flask analyses at all in 2023-2024).  Programmatic
# flasks resumed in 2025, so the two run concurrently again.
_PFP_SITES = {'MLO_PFP': 'MLO', 'MKO_PFP': 'MKO'}


def _base_sites(sites: list[str]) -> list[str]:
    """Map pseudo-site names onto the base sites actually stored in the view."""
    out = []
    for site in sites:
        base = _PFP_SITES.get(site.upper(), site).upper()
        if base not in out:
            out.append(base)
    return out


def _site_label_sql(column: str = 'site', pair_id: str = 'pair_id_num') -> str:
    """SQL CASE expression relabelling PFP rows onto their pseudo-site name.

    Rows of a base site with no pair_id_num are CCGG event (PFP) flasks; every
    other row keeps its own site code.
    """
    whens = ' '.join(
        f"WHEN UPPER({column}) = '{base}' AND {pair_id} = 0 THEN '{pseudo}'"
        for pseudo, base in _PFP_SITES.items()
    )
    return f'CASE {whens} ELSE UPPER({column}) END'


HEADER_FILE = Path(__file__).parent / 'mstar_header.txt'
# Column-description blocks substituted into {columns} in mstar_header.txt.
COLUMNS_PAIRS_FILE = Path(__file__).parent / 'mstar_columns_pairs.txt'
COLUMNS_MONTHLY_FILE = Path(__file__).parent / 'mstar_columns_monthly.txt'
MISSING = 'nd'
MONTHLY_MISSING = 'nan'
# Column order for the global-means export: means before the site columns.
_MEAN_COL_ORDER = ['Global', 'NH', 'SH', 'HN', 'LN', 'LS', 'HS']
# The global-means product follows GML_means' 3-decimal convention; 2 would
# round away the propagated uncertainties, which are often a few hundredths.
GLOBAL_MEANS_DECIMALS = 3
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
    COLUMNS_FILE = COLUMNS_PAIRS_FILE

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
        """Fill the {columns}, {filename} and {date} placeholders in mstar_header.txt."""
        template = HEADER_FILE.read_text()
        return (template
                .replace('{columns}', self.COLUMNS_FILE.read_text().rstrip('\n'))
                .replace('{filename}', filename)
                .replace('{date}', datetime.now().strftime('%Y-%m-%d')))

    # ── data query ───────────────────────────────────────────────────────────

    def query_data(self) -> pd.DataFrame:
        """Query ng_pair_avg_view for all M* instruments, returning a DataFrame.

        PFP rows are relabelled onto their pseudo-site (MLO -> MLO_PFP), so a
        request for MLO returns programmatic flask pairs only.  The relabelled
        code lands in ``export_site``; ``site`` keeps the view's own value.
        """
        insts = ', '.join(f"'{i}'" for i in self.INSTRUMENTS)
        if not self.sites:
            return pd.DataFrame()
        base_list = ', '.join(f"'{s}'" for s in _base_sites(self.sites))
        want_list = ', '.join(f"'{s}'" for s in self.sites)
        # The relabelling has to happen before it can be filtered on, hence the
        # subquery: MySQL cannot reference a SELECT alias from WHERE.
        sql = f"""
        SELECT * FROM (
            SELECT v.*, {_site_label_sql('v.site', 'v.pair_id_num')} AS export_site
            FROM hats.ng_pair_avg_view v
            WHERE v.inst_id IN ({insts})
              AND v.parameter_num = %s
              AND UPPER(v.site) IN ({base_list})
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
        ) t
        WHERE t.export_site IN ({want_list})
        ORDER BY t.export_site, t.sample_datetime
        """
        rows = self.instrument.doquery(sql, [self.parameter_num, self.start_year, self.end_year])
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df['sample_datetime'] = pd.to_datetime(df['sample_datetime'])
            df = df.sort_values(['export_site', 'sample_datetime'])
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
                str(row.get('export_site', row['site'])).lower(),
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


class MstarMonthlyExporter(MstarDataExporter):
    """Export M-system (M1/M3/M4) monthly means of flask pair averages.

    Same single-file, tab-separated layout as :class:`MstarDataExporter`, but
    each row is one calendar month at one site.  Months run continuously from
    each site's first to its last sampled month; a month with no accepted
    flask pair is written as ``nan`` with ``n = 0``.
    """

    COLUMNS_FILE = COLUMNS_MONTHLY_FILE

    # ── data query ───────────────────────────────────────────────────────────

    def query_data(self) -> pd.DataFrame:
        """Return per-site monthly means of pair_avg, aggregated in the DB.

        Columns: site, month_start, monthly_avg, monthly_std, monthly_n.
        Months with no data are not returned here; :meth:`fill_month_gaps`
        inserts them.
        """
        insts = ', '.join(f"'{i}'" for i in self.INSTRUMENTS)
        if not self.sites:
            return pd.DataFrame()
        base_list = ', '.join(f"'{s}'" for s in _base_sites(self.sites))
        want_list = ', '.join(f"'{s}'" for s in self.sites)
        # PFP rows are grouped under their own pseudo-site, so a site's monthly
        # mean never blends programmatic flask pairs with PFP pairs.
        sql = f"""
        SELECT export_site AS site, month_start,
               AVG(pair_avg)    AS monthly_avg,
               STDDEV(pair_avg) AS monthly_std,
               COUNT(pair_avg)  AS monthly_n
        FROM (
            SELECT {_site_label_sql('v.site', 'v.pair_id_num')} AS export_site,
                   DATE_FORMAT(v.sample_datetime, '%%Y-%%m-01') AS month_start,
                   v.pair_avg
            FROM hats.ng_pair_avg_view v
            WHERE v.inst_id IN ({insts})
              AND v.parameter_num = %s
              AND UPPER(v.site) IN ({base_list})
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
        ) t
        WHERE t.export_site IN ({want_list})
        GROUP BY export_site, month_start
        ORDER BY export_site, month_start
        """
        rows = self.instrument.doquery(sql, [self.parameter_num, self.start_year, self.end_year])
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if df.empty:
            return df
        df['month_start'] = pd.to_datetime(df['month_start'])
        # STDDEV() is NULL for a single-pair month; COUNT() never is.
        df['monthly_n'] = df['monthly_n'].astype(int)
        return self.fill_month_gaps(df)

    @staticmethod
    def fill_month_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """Reindex each site onto a continuous monthly series.

        Gap months get NaN mean/std and n = 0, so the month-year sequence is
        unbroken between each site's first and last sampled month.
        """
        filled = []
        for site, grp in df.groupby('site', sort=True):
            grp = grp.set_index('month_start').sort_index()
            months = pd.date_range(grp.index.min(), grp.index.max(), freq='MS')
            grp = grp.reindex(months)
            grp['site'] = site
            grp['monthly_n'] = grp['monthly_n'].fillna(0).astype(int)
            filled.append(grp.rename_axis('month_start').reset_index())
        return _concat_frames(filled)

    # ── format ───────────────────────────────────────────────────────────────

    @staticmethod
    def month_mid_decimal_year(dt: datetime) -> float:
        """Decimal year of the midpoint of *dt*'s calendar month."""
        start = datetime(dt.year, dt.month, 1)
        end = datetime(dt.year + 1, 1, 1) if dt.month == 12 \
            else datetime(dt.year, dt.month + 1, 1)
        mid = start + (end - start) / 2
        return MstarDataExporter.decimal_year(mid)

    def format_lines(self, df: pd.DataFrame) -> list[str]:
        """Return a list of tab-separated data lines (including the column header)."""
        col_mf = self.parameter
        col_line = '\t'.join([
            'site', 'yyyy', 'mm', 'dec_date', col_mf, f'{col_mf}_sd', f'{col_mf}_n',
        ])
        lines = [col_line]
        for _, row in df.iterrows():
            dt = row['month_start']
            if not isinstance(dt, datetime):
                dt = pd.Timestamp(dt).to_pydatetime()

            mf = row['monthly_avg']
            sd = row['monthly_std']
            n = int(row['monthly_n'])
            mf_str = MONTHLY_MISSING if mf is None or mf != mf else f'{mf:.{MF_DECIMALS}f}'
            # STDDEV() of a single pair is 0, which would read as perfect
            # agreement rather than "no spread measurable".
            sd_str = (MONTHLY_MISSING if n < 2 or sd is None or sd != sd
                      else f'{sd:.{SD_DECIMALS}f}')

            lines.append('\t'.join([
                str(row['site']).lower(),
                f'{dt.year}',
                f'{dt.month}',
                f'{self.month_mid_decimal_year(dt):.5f}',
                mf_str,
                sd_str,
                str(n),
            ]))
        return lines

    # ── export ────────────────────────────────────────────────────────────────

    def default_filename(self) -> str:
        """Return a sensible default output filename."""
        return f'{self.parameter}_GCMS_flasks_monthly.txt'


class MstarGlobalMeansExporter(MstarMonthlyExporter):
    """Export monthly hemispheric and global means of M-system flask pair data.

    One row per month holding the global, hemispheric and semi-hemispheric
    means with propagated uncertainties, followed by the monthly mean, standard
    deviation and pair count of every background site they are built from.
    The site list and the weighting rules come from
    ``gml_global_means_config.yaml``; the Timeseries site checkboxes are not
    consulted, so the file always contains everything behind the means.

    See :mod:`global_means` for the math.
    """

    def __init__(self, instrument, parameter: str, parameter_num: int,
                 start_year: int, end_year: int,
                 config: 'GlobalMeansConfig | None' = None):
        self.config = config or GlobalMeansConfig.load()
        self.calculator = GlobalMeansCalculator(parameter, config=self.config)
        # Populated by query_data(); reported in the file header.
        self.sites_without_data: list[str] = []
        super().__init__(
            instrument=instrument,
            parameter=parameter,
            parameter_num=parameter_num,
            sites=self.config.sites_for(parameter),
            start_year=start_year,
            end_year=end_year,
        )

    # ── site metadata ────────────────────────────────────────────────────────

    def site_latitudes(self) -> dict[str, float]:
        """Return {site: lat} for the configured background sites.

        PFP pseudo-sites are absent from gmd.site, so they inherit their base
        site's latitude -- the PFPs sample the same place.  That does mean a
        location running both flask and PFP concurrently carries twice the
        weight of a single-programme site in its band; see the note in
        gml_global_means_config.yaml.
        """
        if not self.sites:
            return {}
        base = _base_sites(self.sites)
        placeholders = ', '.join(['%s'] * len(base))
        rows = self.instrument.doquery(
            f'SELECT code, lat FROM gmd.site WHERE code IN ({placeholders})', base
        )
        lats = {r['code'].lower(): float(r['lat']) for r in rows or []}
        for pseudo, base_site in _PFP_SITES.items():
            if base_site.lower() in lats:
                lats[pseudo.lower()] = lats[base_site.lower()]
        return lats

    # ── data query ───────────────────────────────────────────────────────────

    def query_site_months(self) -> pd.DataFrame:
        """Per-site monthly means of the M* flask pair averages.

        ``sd`` is the spread of the pair means in the month, falling back to the
        single pair's own within-pair standard deviation when only one pair was
        collected — a month of one pair has no spread of its own, and the
        propagated uncertainties need a value there.
        """
        insts = ', '.join(f"'{i}'" for i in self.INSTRUMENTS)
        if not self.sites:
            return pd.DataFrame()
        base_list = ', '.join(f"'{s}'" for s in _base_sites(self.sites))
        want_list = ', '.join(f"'{s}'" for s in self.sites)
        sql = f"""
        SELECT LOWER(export_site) AS site, date,
               AVG(pair_avg)    AS mf,
               STDDEV(pair_avg) AS sd_spread,
               AVG(pair_stdv)   AS sd_pair,
               COUNT(pair_avg)  AS n
        FROM (
            SELECT {_site_label_sql('v.site', 'v.pair_id_num')} AS export_site,
                   DATE_FORMAT(v.sample_datetime, '%%Y-%%m-01') AS date,
                   v.pair_avg, v.pair_stdv
            FROM hats.ng_pair_avg_view v
            WHERE v.inst_id IN ({insts})
              AND v.parameter_num = %s
              AND UPPER(v.site) IN ({base_list})
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
        ) t
        WHERE t.export_site IN ({want_list})
        GROUP BY export_site, date
        ORDER BY date, export_site
        """
        rows = self.instrument.doquery(sql, [self.parameter_num, self.start_year, self.end_year])
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if df.empty:
            return df
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        for col in ('mf', 'sd_spread', 'sd_pair'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['n'] = df['n'].astype(int)
        df['sd'] = df['sd_spread'].where(df['n'] >= 2, df['sd_pair'])
        return df[['site', 'date', 'mf', 'sd', 'n']]

    def query_data(self) -> pd.DataFrame:
        """Return the assembled means frame, means first then the site columns."""
        site_df = self.query_site_months()
        if site_df.empty:
            return pd.DataFrame()

        lats = self.site_latitudes()
        # Site columns come from the same gap-filled frame the means used, so an
        # interpolated month shows the value that fed them, with n = 0.
        prepared = self.calculator.prepare(site_df, lats)
        means = self.calculator.compute_prepared(prepared)
        if means.empty:
            return pd.DataFrame()

        # Configured sites that yielded nothing -- absent from gmd.site, filtered
        # as a PFP pseudo-site, or simply never run on an M-system instrument.
        self.sites_without_data = sorted(
            {s.lower() for s in self.sites} - set(prepared['site'])
        )

        wide = prepared.pivot_table(index='date', columns='site',
                                    values=['mf', 'sd', 'n'], sort=True)
        site_cols = {}
        for site in sorted({s for _, s in wide.columns}):
            site_cols[site] = wide[('mf', site)]
            site_cols[f'{site}_sd'] = wide[('sd', site)]
            site_cols[f'{site}_n'] = wide[('n', site)]
        sites_df = pd.DataFrame(site_cols)

        out = means.join(sites_df, how='outer')
        out.index.name = 'date'
        return out.reset_index()

    # ── header ────────────────────────────────────────────────────────────────

    def build_header(self, filename: str) -> str:
        """Fill the config's global-means header template."""
        skipped = sorted(set(self.sites_without_data))
        used = [s.lower() for s in self.sites if s.lower() not in skipped]
        skipped_note = (
            f'# Listed background sites with no M-system data, omitted:\n'
            f'#   {", ".join(skipped)}' if skipped else '#'
        )
        interp_note = self._interpolation_note()
        psa_lat = self.calculator.weight_lats.get('psa')
        psa_note = (
            f'\n#   For {self.parameter} PSA is also moved, weighted as '
            f'{abs(psa_lat):.0f}S instead of its\n#   true ~64S.' if psa_lat else ''
        )
        source_note = (
            '#\n#   Note: GML publishes this gas from blended fECD and MSD\n'
            '#   measurements.  This file is M-system only, so it will not\n'
            '#   reproduce the published global means exactly.'
            if self.config.is_combined_source(self.parameter) else '#'
        )
        phi = f'{self.config.phi:g}'
        return (self.config.header_template
                .replace('{filename}', filename)
                .replace('{param}', self.parameter)
                .replace('{generated_on}', datetime.now().strftime('%Y-%m-%d'))
                .replace('{phi}', phi)
                .replace('{background_sites}', ', '.join(used))
                .replace('{skipped_sites}', skipped_note)
                .replace('{psa_note}', psa_note)
                .replace('{interp_note}', interp_note)
                .replace('{source_note}', source_note))

    def _interpolation_note(self) -> str:
        """Comment block describing how gaps were filled, per the active config."""
        if not self.config.interpolate_site_gaps:
            return ('#   Months with no accepted flask pair are left blank; nothing\n'
                    '#   is interpolated or modelled.')
        cap = self.config.max_interpolation_months
        if self.config.interpolation_method == 'seasonal':
            how = ('#   Interior gaps in a site\'s record are filled from an additive\n'
                   '#   Holt-Winters fit of that site\'s own series -- level, trend and\n'
                   '#   a 12-month seasonal term -- taking the model value only where an\n'
                   '#   observation is missing; observed months are never replaced.  A\n'
                   '#   site with fewer than two full seasonal cycles falls back to\n'
                   '#   linear interpolation.  The standard deviation is interpolated in\n'
                   '#   time rather than modelled.')
        else:
            how = ('#   Interior gaps in a site\'s record are filled by linear\n'
                   '#   interpolation in time.')
        note = (f'{how}\n'
                '#   Filled months carry n = 0.  A run of more than '
                f'{cap} consecutive\n'
                '#   missing months is left empty rather than bridged, so a long\n'
                '#   outage is never inferred.')
        # MLO's programmatic flask record has a multi-year hole that mlo_pfp
        # covers; without this the blank mlo column looks like an error.
        if any(s.upper() == 'MLO_PFP' for s in self.sites):
            note += (
                '\n#\n'
                '#   mlo and mlo_pfp are separate sites here.  PFPs (programmable\n'
                '#   flask packages) are a different kind of flask, deployed at MLO\n'
                '#   in 2021 and the only sampling there from Dec 2022 to Aug 2025\n'
                '#   after the Nov 2022 eruption cut power and access to the\n'
                '#   observatory.  mlo is blank over that outage by design: the gap\n'
                '#   is too long to bridge, and mlo_pfp carries the record.'
            )
        return note

    # ── format ───────────────────────────────────────────────────────────────

    def format_lines(self, df: pd.DataFrame) -> list[str]:
        """Return tab-separated data lines, means columns first then per-site."""
        mean_cols, site_cols = [], []
        for col in df.columns:
            if col == 'date':
                continue
            (mean_cols if col.split('_')[0] in _MEAN_COL_ORDER else site_cols).append(col)
        mean_cols.sort(key=lambda c: (_MEAN_COL_ORDER.index(c.split('_')[0]),
                                      c.endswith('_sd')))

        lines = ['\t'.join(['yyyy', 'mm', 'dec_date'] + mean_cols + site_cols)]
        for _, row in df.iterrows():
            dt = row['date']
            if not isinstance(dt, datetime):
                dt = pd.Timestamp(dt).to_pydatetime()
            fields = [f'{dt.year}', f'{dt.month}',
                      f'{self.month_mid_decimal_year(dt):.5f}']
            for col in mean_cols + site_cols:
                val = row.get(col)
                if col.endswith('_n'):
                    fields.append('0' if val is None or val != val else str(int(val)))
                elif val is None or val != val:
                    fields.append(MONTHLY_MISSING)
                else:
                    fields.append(f'{val:.{GLOBAL_MEANS_DECIMALS}f}')
            lines.append('\t'.join(fields))
        return lines

    # ── export ────────────────────────────────────────────────────────────────

    def default_filename(self) -> str:
        return f'{self.parameter}_GCMS_global_means.txt'

    def export(self, output_path: str | Path) -> int:
        """Query, compute and write.  Returns the number of monthly records."""
        df = self.query_data()
        if df.empty:
            return 0
        # format_lines first: build_header reports the sites the means actually
        # used, which prepare() only records once it has run.
        lines = self.format_lines(df)
        header = self.build_header(Path(output_path).name)
        Path(output_path).write_text(header + '\n'.join(lines) + '\n')
        return len(lines) - 1

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_timeseries_widget(
        cls, widget, sites: list[str] | None = None, all_time: bool = False
    ) -> 'MstarGlobalMeansExporter':
        """Construct from a TimeseriesWidget.

        *sites* is accepted for signature compatibility with the sibling
        exporters and ignored: the background sites come from the config.
        """
        analyte = widget.analyte_combo.currentText()
        return cls(
            instrument=widget.instrument,
            parameter=analyte,
            parameter_num=widget.analytes.get(analyte),
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
        SELECT sample_datetime, pair_avg, pair_stdv, pair_id_num, '' AS sample_type,
               'OTTO' AS instrument
        FROM hats.ng_pair_avg_view
        WHERE inst_id = 'OTTO'
          AND parameter_num = %s
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
            SELECT sample_datetime, pair_avg, pair_stdv, pair_id_num, '' AS sample_type,
                   'FE3' AS instrument
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
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
            SELECT sample_datetime, pair_avg, pair_stdv, pair_id_num, '' AS sample_type,
                   'FE3' AS instrument
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
              AND channel = %s
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
            SELECT sample_datetime, pair_avg, pair_stdv, pair_id_num, '' AS sample_type,
                   'FE3' AS instrument
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
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
