"""Flask/pair-analysis instrument classes: M4, FE3, and Perseus.
Import public names through the logos_instruments facade."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from functools import cached_property
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time

from logos_instruments_core import HATS_DB_Functions, Normalizing

class M4_Instrument(HATS_DB_Functions):
    """ Class for accessing M4 specific functions in the HATS database. """

    RUN_TYPE_MAP = {
        "All": None,        # no filter
        "Flasks": 1,        # run_type_num
        #"Calibrations": 2,
        "PFPs": 5,
    }
    STANDARD_RUN_TYPE = 8
    CAL_RUN_TYPES = {7}  # run_type_num values written to hats.calibrations
    EXCLUDE = [6, 7]     # run_type_num to exclude from autoscaling (zero air and tank runs)
    # Require >=3 unrejected injections for a calibration row, matching FE3.
    # A num=1/2 group has too few injections for a meaningful stddev and
    # can silently skew a caldrift fit even though it's excluded from the
    # plotted calibration series (which also requires num >= 3).
    MIN_CAL_INJECTIONS = 3
    
    MARKER_MAP = {
        # run_type_num
        1: 'o',   # Flask
        5: 's',   # PFP
        6: 'v',   # Zero
        7: '^',   # Tank
        8: 'D',   # Standard
    }

    _MISSING_LABEL_VALUES = {'', '0', 'nan', 'none', '<na>'}
    
    def __init__(self):
        super().__init__()
        self.inst_id = 'm4'
        self.inst_num = self.INSTRUMENTS.get(self.inst_id)  # Lookup inst_num from INSTRUMENTS
        self.start_date = '19940718'         # M-system tank/calibration history starts with M3 records.
        self.gc_dir = Path("/hats/gc/m4")
        self.export_dir = self.gc_dir / "results"

        self.molecules = self.query_molecules()
        self.analytes = self.query_analytes()
        self.analytes_inv = {int(v): k for k, v in self.analytes.items()}
        self.response_type = 'area'
        
        self.norm = Normalizing(self.inst_id, self.STANDARD_RUN_TYPE, 'run_type_num', self.response_type)
        self.calibration_inst_ids = ('M1', 'm1', 'M3', 'm3', 'M4')
                
    def load_data(self, pnum, channel=None, run_type_num=None, start_date=None, end_date=None, verbose=True):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            run_type_num (int, optional): Run type number to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        if end_date is None:
            end_date = datetime.today()
        elif len(end_date) == 4:
            # check for YYMM format
            end_date = datetime.strptime(end_date, "%y%m")
            end_date = end_date.strftime("%Y-%m-31")    # end of the month
        else:
            # expecting '%Y-%m-%d %H:%M:%s' format
            pass

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif len(start_date) == 4: 
            # check for YYMM format
            start_date = datetime.strptime(start_date, "%y%m")
            start_date = start_date.strftime("%Y-%m-01")    # beginning of the month
        else:
            # expecting '%Y-%m-%d %H:%M:%s' format
            pass
       
        # the run_type_num for M4 is not the same for all run_times
        # Don't use for filtering 
        if run_type_num is not None:
            run_type_filter = f"AND run_type_num = {run_type_num}"

        if verbose:
            print(f"Loading data from {start_date} to {str(end_date)} for parameter {pnum}")
        # todo: use flags - using low_flow flag
        query = f"""
            SELECT * FROM hats.ng_data_processing_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                AND run_time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            if verbose:
                print(f"No data found for parameter {pnum} in the specified date range.")
            return pd.DataFrame()
        
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'], errors='raise', utc=True)
        df['run_time']          = pd.to_datetime(df['run_time'], errors='raise', utc=True)
        df['sample_datetime']   = pd.to_datetime(df['sample_datetime'], errors='raise', utc=True)
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].astype(int)
        df['port']              = df['port'].astype(int)
        df['flask_port']        = df['flask_port'].astype(int, errors='ignore')  # handle NaN gracefully
        df['area']              = df['area'].astype(float)
        df['net_pressure']      = df['net_pressure'].astype(float)
        df['area']              = df['area']/df['net_pressure']         # response per pressure
        df['mole_fraction']     = df['mole_fraction'].astype(float)
        df = self.norm.merge_smoothed_data(df)
        df['parameter_num']     = pnum

        if 'rejected' in df.columns:
            df['rejected'] = df['rejected'].fillna(0).astype(int)
        else:
            df['rejected'] = 0
        df.drop(columns=['data_flag'], errors='ignore', inplace=True)
        
        # build a port_idx for plotting colors
        mask = df['run_type_num'].eq(5)     # pfp runtype
        base = pd.to_numeric(df['port'], errors='coerce').astype('float64')
        pfp  = pd.to_numeric(df['flask_port'], errors='coerce').astype('float64') + 20
        res = base.copy()
        res.loc[mask] = pfp.loc[mask]          # explicit assignment avoids where/mask downcast warning
        df['port_idx'] = res.round().astype('Int64')   # final, intentional cast to nullable int

        # Keep PFP packages distinct in the legend.  sample_id/site can be
        # missing for a second PFP, while port_info still contains e.g.
        # ``5-3129`` (flask port 5, package 3129).
        if mask.any():
            flask_ports = pd.to_numeric(df.loc[mask, 'flask_port'], errors='coerce').fillna(-1).astype(int)
            pfp_ids = self._pfp_sample_identity(df.loc[mask])
            combos = pd.Series(list(zip(flask_ports, pfp_ids)), index=df.index[mask])
            pfp_codes = pd.Series(pd.factorize(combos)[0] + 200, index=df.index[mask])  # offset to avoid collisions with real ports
            df.loc[mask, 'port_idx'] = pfp_codes.astype('Int64')
        
        df = self.add_port_labels(df)       # port labels, colors, and markers
        
        return df.sort_values('analysis_datetime')

    @classmethod
    def _clean_label_text(cls, values):
        """Return stripped strings with database placeholder values as NA."""
        cleaned = values.astype('string').str.strip()
        return cleaned.mask(cleaned.str.lower().isin(cls._MISSING_LABEL_VALUES))

    @classmethod
    def _pfp_sample_identity(cls, pfp_df):
        """Return ``package-flask`` IDs, falling back to M4 ``port_info``.

        Fully associated PFP rows already carry sample_id values such as
        ``3930-05``.  When event metadata is unavailable, M4 still records the
        same information in the opposite order in port_info (``5-3129``).
        """
        sample_ids = cls._clean_label_text(pfp_df['sample_id'])
        port_info = cls._clean_label_text(pfp_df['port_info'])
        parsed = port_info.str.extract(
            r'^(?P<flask_port>\d{1,2})-(?P<package>\d{4}|[xX]{4})$'
        )
        from_port_info = (
            parsed['package'].str.upper() + '-' +
            parsed['flask_port'].str.zfill(2)
        )
        return sample_ids.fillna(from_port_info).fillna(port_info).fillna('PFP')

    @staticmethod
    def _darken_color(color, step):
        """Return a progressively darker variant while preserving opacity."""
        rgba = np.asarray(to_rgba(color), dtype=float)
        factor = max(0.55, 1.0 - (0.18 * step))
        return tuple(np.append(rgba[:3] * factor, rgba[3]))

    def _apply_pfp_package_colors(self, df, mask, site_colors):
        """Shade later PFP packages within the same M4 run and site."""
        if not mask.any():
            return

        pfp = df.loc[mask].copy()
        identities = self._pfp_sample_identity(pfp)
        pfp['_package'] = identities.str.extract(
            r'^(?P<package>[^-]+)-\d{1,2}$', expand=False
        )
        pfp['_site'] = self._clean_label_text(pfp['site'])
        if 'run_time' in pfp:
            pfp['_run'] = pfp['run_time'].astype('string').fillna('__unknown_run__')
        else:
            pfp['_run'] = '__single_run__'
        if 'analysis_datetime' in pfp:
            pfp['_order'] = pd.to_datetime(
                pfp['analysis_datetime'], errors='coerce', utc=True
            )
        else:
            pfp['_order'] = pd.RangeIndex(len(pfp))
        pfp['_row_order'] = np.arange(len(pfp))

        for (_run, site), group in pfp.dropna(subset=['_site']).groupby(
            ['_run', '_site'], sort=False
        ):
            package_order = (
                group.sort_values(['_order', '_row_order'])['_package']
                .dropna()
                .drop_duplicates()
                .tolist()
            )
            if len(package_order) < 2:
                continue
            base_color = site_colors.get(site, 'gray')
            for shade_step, package in enumerate(package_order[1:], start=1):
                indices = group.index[group['_package'].eq(package)]
                color = self._darken_color(base_color, shade_step)
                df.loc[indices, 'port_color'] = pd.Series(
                    [color] * len(indices), index=indices
                )
        
    def add_port_labels(self, df):
        """ Helper function to add port labels to the dataframe. """
        
        # base port label on port_info and port number
        df['port_label'] = (
            df['port_info'].fillna('').str.strip() + ' (' +
            df['port'].astype(int).astype(str) + ')'
        )

        # flask_port label
        mask = (df['run_type_num'] == 1)
        df.loc[mask, 'port_label'] = (
            df.loc[mask, 'site'] + ' ' +
            df.loc[mask, 'pair_id_num'].astype(int).astype(str) + '-' +
            df.loc[mask, 'sample_id'].astype(int).astype(str) + ' (' +
            df.loc[mask, 'port'].astype(int).astype(str) + ')'
        )

        # pfp label
        mask = (df['run_type_num'] == 5)
        if mask.any():
            pfp_df = df.loc[mask]
            sample_ids = self._pfp_sample_identity(pfp_df)
            sites = self._clean_label_text(pfp_df['site'])
            flask_ports = pd.to_numeric(
                pfp_df['flask_port'], errors='coerce'
            ).astype('Int64').astype('string')
            labels = sample_ids.copy()
            labels.loc[sites.notna()] = sites.loc[sites.notna()] + ' ' + labels.loc[sites.notna()]
            df.loc[mask, 'port_label'] = labels + ' (' + flask_ports.fillna('?') + ')'

        # clean up any stray spaces
        df['port_label'] = df['port_label'] \
                            .str.replace(r'\s+', ' ', regex=True) \
                            .str.strip()

        # assign colors to sites
        cmap = plt.get_cmap('tab20')
        site_colors = {site: cmap(i % 20) for i, site in enumerate(self.LOGOS_sites)}

        # Start with site-based colors
        df['port_color'] = df['site'].map(site_colors).fillna('gray')

        # M4 can analyze multiple PFP packages from the same site in one run.
        # Preserve the site's base color for the first package and shade later
        # packages so their points and legend entries remain distinguishable.
        self._apply_pfp_package_colors(df, mask, site_colors)

        # Override when port == 14 → gray
        df.loc[df['run_type_num'] == self.STANDARD_RUN_TYPE, 'port_color'] = 'red'

        # Override when port_info == 'zero_air' → black
        df.loc[df['port_info'] == 'zero_air', 'port_color'] = 'black'  
        
        df['port_marker'] = df['run_type_num'].map(self.MARKER_MAP).fillna('o')

        return df            

    def _calc_mole_fraction_cfc113a_single(self, df):
        """
        Called by calc_mole_fraction() when pnum is 32 or 178.
        df contains freshly-normalised data for ONE of the two parameters.
        The partner parameter is loaded from the DB so the Montzka deconvolution
        can be solved, and only the relevant mole_fraction column is returned.

        The partner uses DB-stored normalized_resp values (i.e. whatever was last
        written by load_data / upsert). This is the correct behaviour for flagging
        and batch recalc. For smoothing changes, the partner may momentarily be
        slightly stale until the user re-loads that parameter — acceptable given
        that both parameters must ultimately be saved separately anyway.
        """
        if df.empty:
            out = df.copy()
            out['mole_fraction'] = pd.Series(dtype=float)
            return out

        pnum = int(df['parameter_num'].iat[0])
        partner_pnum = 178 if pnum == 32 else 32

        # Date window covering all rows in df
        t_min = df['run_time'].min()
        t_max = df['run_time'].max()
        start = (t_min - pd.Timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
        end   = (t_max + pd.Timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')

        partner_df = self.load_data(partner_pnum, start_date=start, end_date=end, verbose=False)

        if partner_df.empty:
            out = df.copy()
            out['mole_fraction'] = np.nan
            return out

        # Merge the caller's df (fresh normalization) with partner (from DB)
        if pnum == 32:
            left, right = df, partner_df
        else:
            left, right = partner_df, df

        merged = left.merge(
            right[['analysis_num', 'normalized_resp', 'detrend_method_num', 'channel']],
            on='analysis_num',
            suffixes=('_32', '_178'),
            how='inner'
        )

        solved = self.calc_mole_fraction_cfc113a(merged)

        # Pick the mole_fraction column for the requested pnum
        mf_col = 'mole_fraction_cfc113' if pnum == 32 else 'mole_fraction_cfc113a'
        mf = solved.set_index('analysis_num')[mf_col]

        out = df.copy()
        out['mole_fraction'] = out['analysis_num'].map(mf)
        return out

    def cfc113a_response_factors(self, run_date):
        """
        Return the molar response factors R1–R4 from hats.ng_cfc113a
        for the row whose date window contains run_date.

        Args:
            run_date: datetime, date, or Timestamp of the GC run.
        Returns:
            dict with keys R1, R2, R3, R4, or None if no row matches.
        """
        if hasattr(run_date, 'date'):
            run_date = run_date.date()
        result = self.doquery(
            """
            SELECT R1, R2, R3, R4
            FROM hats.ng_cfc113a
            WHERE inst_num = %s
              AND datetime_start <= %s
              AND (datetime_stop > %s OR datetime_stop IS NULL)
            ORDER BY num DESC
            LIMIT 1
            """,
            [self.inst_num, run_date, run_date]
        )
        return result[0] if result else None

    def load_data_cfc113a(self, start_date=None, end_date=None):
        """
        Load CFC-113 (pnum=32, ion 103) and CFC-113a (pnum=178, ion 117)
        data for the same date range, normalize each independently against
        their respective smoothed reference-gas curves, then merge on
        analysis_num.

        Returns:
            DataFrame with all columns from the pnum=32 load, plus
            normalized_resp_32, normalized_resp_178 (both already
            pressure-corrected and ref-gas-smoothed).
            Returns an empty DataFrame if either parameter has no data.
        """
        df_32  = self.load_data(32,  start_date=start_date, end_date=end_date)
        df_178 = self.load_data(178, start_date=start_date, end_date=end_date)

        if df_32.empty or df_178.empty:
            return pd.DataFrame()

        df = df_32.merge(
            df_178[['analysis_num', 'normalized_resp', 'detrend_method_num', 'channel']],
            on='analysis_num',
            suffixes=('_32', '_178'),
            how='inner'
        )
        return df

    def calc_mole_fraction_cfc113a(self, df):
        """
        Solve the Montzka (Jan 2026) two-compound simultaneous equations
        to deconvolve CFC-113 and CFC-113a from their overlapping ion signals.

        Ion 103 (pnum=32):  RX = MFA*R1 + MFB*R2
        Ion 117 (pnum=178): RY = MFA*R3 + MFB*R4

        Solving:
            MFA = (RX - RY * R2/R4) / (R1 - R3 * R2/R4)     [CFC-113]
            MFB = (RY - MFA * R3) / R4                      [CFC-113a]

        R1–R4 are fetched from hats.ng_cfc113a by run_time date window.

        Args:
            df: DataFrame returned by load_data_cfc113a(), with columns
                normalized_resp_32 (RX) and normalized_resp_178 (RY).
        Returns:
            df copy with added columns mole_fraction_cfc113 (MFA) and
            mole_fraction_cfc113a (MFB).
        """
        if df.empty:
            out = df.copy()
            out['mole_fraction_cfc113']  = pd.Series(dtype=float)
            out['mole_fraction_cfc113a'] = pd.Series(dtype=float)
            return out

        mf_a = pd.Series(index=df.index, dtype=float)
        mf_b = pd.Series(index=df.index, dtype=float)

        rf_cache = {}

        for rt, grp in df.groupby('run_time'):
            if rt not in rf_cache:
                rf_cache[rt] = self.cfc113a_response_factors(rt)
            rf = rf_cache[rt]

            if rf is None:
                mf_a.loc[grp.index] = np.nan
                mf_b.loc[grp.index] = np.nan
                continue

            R1 = float(rf['R1'])
            R2 = float(rf['R2'])
            R3 = float(rf['R3'])
            R4 = float(rf['R4'])

            RX = grp['normalized_resp_32'].values
            RY = grp['normalized_resp_178'].values

            denom = R1 - R3 * (R2 / R4)
            if abs(denom) < 1e-12:
                mf_a.loc[grp.index] = np.nan
                mf_b.loc[grp.index] = np.nan
                continue

            MFA = (RX - RY * (R2 / R4)) / denom
            MFB = (RY - MFA * R3) / R4

            mf_a.loc[grp.index] = MFA
            mf_b.loc[grp.index] = MFB

        out = df.copy()
        out['mole_fraction_cfc113']  = mf_a
        out['mole_fraction_cfc113a'] = mf_b
        return out

    def upsert_cfc113a_pair(self, df, response_id=None):
        """
        Write CFC-113 and CFC-113a mole fractions from calc_mole_fraction_cfc113a()
        back to hats.ng_mole_fractions as two separate parameter rows per analysis.

        Args:
            df:          DataFrame returned by calc_mole_fraction_cfc113a().
            response_id: Optional ng_response_id to tag both sets of rows.
        """
        # --- CFC-113 (pnum=32) ---
        df_a = df.copy()
        df_a['mole_fraction']      = df_a['mole_fraction_cfc113']
        df_a['parameter_num']      = 32
        df_a['detrend_method_num'] = df_a['detrend_method_num_32']
        df_a['channel']            = df_a['channel_32']
        self.upsert_mole_fractions(df_a, response_id=response_id)

        # --- CFC-113a (pnum=178) ---
        df_b = df.copy()
        df_b['mole_fraction']      = df_b['mole_fraction_cfc113a']
        df_b['parameter_num']      = 178
        df_b['detrend_method_num'] = df_b['detrend_method_num_178']
        df_b['channel']            = df_b['channel_178']
        self.upsert_mole_fractions(df_b, response_id=response_id)

    def upsert_cfc113a_calibrations(self, df):
        """Write deconvolved CFC-113 and CFC-113a tank calibrations.

        ``df`` retains the generic ``mole_fraction`` column from the pnum-32
        source frame.  Select each compound's deconvolved result explicitly
        before invoking the shared calibration aggregator.
        """
        for parameter_num, mole_fraction_col, normalized_resp_col in (
            (32, 'mole_fraction_cfc113', 'normalized_resp_32'),
            (178, 'mole_fraction_cfc113a', 'normalized_resp_178'),
        ):
            df_cal = df.copy()
            df_cal['mole_fraction'] = df_cal[mole_fraction_col]
            # The shared M4 calibration aggregator applies its uncertainty
            # floor in normalized-response space.  The deconvolution frame
            # retains the two channels with explicit suffixes, so expose the
            # channel corresponding to this calibration result.
            df_cal['normalized_resp'] = df_cal[normalized_resp_col]
            self.upsert_calibrations(df_cal, parameter_num)

    def load_calcurves(self, df):
        """ Calcurves are not stored in ng_response for M4. """
        pass

class FE3_Instrument(HATS_DB_Functions):
    """ Class for accessing FE3 specific functions in the HATS database. """
    
    RUN_TYPE_MAP = {
        "All": None,        # no filter
        "Flasks": 1,        # run_type_num
        "Calibrations": 2,
        "Other": 4,
        "Tank": 7,
        #"PFPs": 5,
        "Test": 10
    }
    STANDARD_PORT_NUM = 1       # port number the standard is run on.
    CAL_RUN_TYPES = {2, 4}      # run_type_num values written to hats.calibrations
    # Require >=3 unrejected injections for a calibration row. FE3's "Other"
    # (run_type 4) cal category also carries single-injection test runs (e.g.
    # multi-tank screening runs in 2022); those aggregate to num=1 rows with
    # stddev=0 that are meaningless as calibrations and break drift fitting.
    MIN_CAL_INJECTIONS = 3
    WARMUP_RUN_TYPE = 3         # run type num warmup runs are on.
    EXCLUDE = [9]               # push port - exclude from autoscaling

    # The plumbing on the GC changed, before 20210928-225324
    # Define the cutoff date
    CUTOFF_DATETIME1 = datetime(2021, 9, 28, 22, 32, 24, tzinfo=timezone.utc)
    # A second small change happened on 20210930-230101
    # this cutoff date affected two flask runs. Instead of coding special logic,
    # for CUTOFF_DATETIME2, we changed the port assingments in the database directly. 
    # CUTOFF_DATETIME2 = datetime(2021, 9, 30, 23, 1, 1, tzinfo=timezone.utc)

    MARKER_MAP = {
        1: 'X',   # Standard
        2: '^',   # Tank
        3: 's',   # Other
        4: 'P',   # Other
        5: 'D',   # Other
        6: '*',   # Other
        7: '^',   # Other
        8: 's',   # Other
        9: 'v',   # Push port
        10: 'o',  # Flask
    }
    
    COLOR_MAP = {STANDARD_PORT_NUM: 'red', 2: 'purple', 3: 'blue', 4: 'green', 5: 'lightblue', 6: 'orange'}
    
    def __init__(self):
        super().__init__()
        self.inst_id     = 'fe3'
        self.inst_num    = 193
        self.start_date  = '20191217'
        self.gc_dir      = Path("/hats/gc/fe3")
        self.export_dir  = self.gc_dir / "results"
        self.response_type = 'height'

        # query raw molecule/analyte dicts
        self.molecules  = self.query_molecules()
        raw_analytes    = self.query_analytes()
        self.analytes_inv = {int(v): k for k, v in raw_analytes.items()}

        # rename CFC11/113 to “(a)” and add “(c)”
        self.analytes = {
            (f"{name} (a)" if name in ('CFC11','CFC113') else name): num
            for name, num in raw_analytes.items()
        }
        self.analytes.update({
            'CFC11 (c)': self.analytes['CFC11 (a)'],
            'CFC113 (c)': self.analytes['CFC113 (a)'],
        })
 
        self.norm = Normalizing(self.inst_id, self.STANDARD_PORT_NUM, 'port', self.response_type)

    @cached_property
    def gc_channels(self) -> dict[str, list[str]]:
        """channel → [display_name, …]"""
        df = pd.DataFrame(self.db.doquery(
            f"SELECT display_name, channel FROM hats.analyte_list WHERE inst_num = {self.inst_num}"
        ))
        return df.groupby('channel')['display_name'].apply(list).to_dict()

    @cached_property
    def molecule_channel_map(self) -> dict[str, str]:
        """lowercase molecule → channel"""
        return {
            mol.lower(): ch
            for ch, mols in self.gc_channels.items()
            for mol in mols
        }

    def return_preferred_channel(self) -> pd.DataFrame:
        """Return preferred channel assignments from hats.ng_preferred_channel for this instrument."""
        sql = """
        SELECT inst_num, parameter_num, start_date, channel
        FROM hats.ng_preferred_channel
        WHERE inst_num = %s
        ORDER BY parameter_num, start_date
        """
        return pd.DataFrame(self.db.doquery(sql, (self.inst_num,)))
    
    def load_data(self, pnum, channel=None, run_type_num=None, start_date=None, end_date=None, verbose=True):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            run_type_num (int, optional): Run type number to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        t0 = time.time()
        
        if end_date is None:
            end_date = datetime.today()
        elif len(end_date) == 4:
            # check for YYMM format
            end_date = datetime.strptime(end_date, "%y%m")
            end_date = end_date.strftime("%Y-%m-31")    # end of the month
            end_date = end_date[:19]  # ensure consistent formatting for SQL query
        else:
            # expecting '%Y-%m-%d %H:%M:%s' format
            pass

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif len(start_date) == 4: 
            # check for YYMM format
            start_date = datetime.strptime(start_date, "%y%m")
            start_date = start_date.strftime("%Y-%m-01")    # beginning of the month
            start_date = start_date[:19]  # ensure consistent formatting for SQL query
        else:
            # expecting '%Y-%m-%d %H:%M:%s' format
            pass
       
        # select run type filter (always exclude warmup runs if no filter specified)
        if run_type_num is not None:
            run_type_filter = f"AND run_type_num = {run_type_num}"
        else:
            run_type_filter = f"AND run_type_num != {self.WARMUP_RUN_TYPE}"
            
        channel_str = f"AND channel = '{channel}'" if channel else ""

        if verbose:
            print(f"Loading data from {start_date} to {end_date} for parameter {pnum}")

        query = f"""
            SELECT * FROM hats.ng_data_processing_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                {channel_str}
                {run_type_filter}
                AND (height IS NULL OR height <> -999)
                AND run_time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            if verbose:
                print(f"No data found for parameter {pnum} in the specified date range.")
            return pd.DataFrame()
        
        #print(f"Data loaded in {time.time()-t0:.4f}s, {len(df)} rows.")
        
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'], errors='raise', utc=True)
        df['run_time']          = pd.to_datetime(df['run_time'], errors='raise', utc=True)
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].astype(int)
        df['height']            = df['height'].astype(float)
        df = self.norm.merge_smoothed_data(df)
        
        # only set scale_num if first row is null
        if df['cal_scale_num'].isna().iat[0]:
            df['cal_scale_num'] = self.qurey_return_scale_num(pnum)

        if 'rejected' in df.columns:
            df['rejected'] = df['rejected'].fillna(0).astype(int)
        else:
            df['rejected'] = 0
        df.drop(columns=['data_flag'], errors='ignore', inplace=True)
        
        df = self.add_port_labels(df)   # port labels, colors, and markers (port_idx)

        return df.sort_values('analysis_datetime')
            
    def add_port_labels(self, df):
        """ Helper function to add port labels to the dataframe. """
        # base port label on port_info and port number
        df['port_label'] = (
            df['port_info'].fillna('unknown').str.strip() + ' (' +
            df['port'].astype(int).astype(str) + ')'
        )

        # flask_port label: if site is NaN, use port_info instead
        mask = df['flask_port'].notna()
        # create a "prefix" that is site when present, otherwise port_info
        prefix = (
            df.loc[mask, 'site']
            .fillna(df.loc[mask, 'port_info'])
            .str.strip()
        )
        # now concatenate everything off that prefix
        df.loc[mask, 'port_label'] = (
            prefix + ' ' +
            #df.loc[mask, 'port_info'].fillna('') +
            df.loc[mask, 'pair_id_num'].fillna(0).astype(int).astype(str) + '-' +
            df.loc[mask, 'sample_id'].fillna(0).astype(int).astype(str) + ' (' +
            df.loc[mask, 'flask_port'].astype(int).astype(str) + ')'
        )

        # clean up any stray spaces
        df['port_label'] = (
            df['port_label']
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        
        df['port_label'] = (
            df['port_label']
            .str.replace('0-0 (0)', '')
            .str.strip()
        )

        # choose which marker map to use based on the cutoff date
        before_mask = df['analysis_datetime'] < self.CUTOFF_DATETIME1
        #after_mask = ~before_mask

        # base port_idx logic for both
        df['port_idx'] = df['port'].astype(int) + df['flask_port'].fillna(0).astype(int)

        # override port_idx for special cases before cutoff
        df.loc[before_mask & (df['port'] == 10), 'port_idx'] = 9   # push port before cutoff
        df.loc[before_mask & (df['port'] == 2),  'port_idx'] = (10 + df['flask_port'].fillna(0).astype(int))

        # assign colors to sites
        cmap = plt.get_cmap('tab20')
        site_colors = {site: cmap(i % 20) for i, site in enumerate(self.LOGOS_sites)}

        # Start with site-based colors
        df['port_color'] = df['site'].map(site_colors).fillna('gray')
        # override with port colors based on port_idx
        df['port_color'] = df['port_idx'].map(self.COLOR_MAP).fillna(df['port_color'])

        # Override when port_info == 'zero_air' → black
        df.loc[df['port_info'] == 'zero_air', 'port_color'] = 'black'  
        
        # port markers
        df['port_marker'] = df['port_idx'].map(self.MARKER_MAP)

        # assign flask markers (port_idx >= 10) to whatever MARKER_MAP[10] is
        flask_marker = self.MARKER_MAP.get(10, 'o')  # fallback just in case
        df.loc[df['port_idx'] >= 10, 'port_marker'] = flask_marker        

        return df
            
    def load_calcurves(self, pnum, channel, selected_run):
        """
        Returns the calibration curves from ng_response for a given parameter number and channel.
        Only curves within 60 days before and 7 days after the selected run date are returned.
        The function applies the following logic to determine the function index:
        - If coef3 == 0 and coef2 == 0 and abs(coef1) > 0 → func_index = 0
        - If coef3 == 0 and abs(coef2) > 0 → func_index = 1
        - If abs(coef3) > 0 → func_index = 2
        """
        scale_num = self.qurey_return_scale_num(pnum)
        
        ch = ''
        if channel is not None:
            ch = f"and channel = '{channel}'"
        
        sql = f"""
            SELECT id, run_date, serial_number, coef0, coef1, coef2, coef3, function, flag FROM hats.ng_response
                where inst_num = {self.inst_num}
                and scale_num = {scale_num}
                {ch}
                and run_date BETWEEN DATE_SUB('{selected_run}', INTERVAL 60 DAY) AND DATE_ADD('{selected_run}', INTERVAL 7 DAY)
            order by run_date desc;
        """
        df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            print(f"No calibration curves found for parameter {pnum} and channel '{channel}'.")
            return df
        df['flag'] = pd.to_numeric(df['flag'], errors='coerce').fillna(1).astype(int)

        df['func_index'] = None  # default

        # Condition 2: coef3 == 0 and abs(coef2) > 0 → func_index = 1
        df.loc[(df['coef3'] == 0) & (df['coef2'].abs() > 0), 'func_index'] = 1

        # Condition 3: abs(coef3) > 0 → func_index = 2
        df.loc[df['coef3'].abs() > 0, 'func_index'] = 2

        # Condition 1: coef3 == 0 and coef2 == 0 and abs(coef1) > 0 → func_index = 0
        df.loc[(df['coef3'] == 0) & (df['coef2'] == 0) & (df['coef1'].abs() > 0), 'func_index'] = 0

        return df

class Perseus_Instrument(HATS_DB_Functions):
    """Combined Perseus PR1/PR2 instrument facade.

    Serves the tank/calibration tools and read-only Processing-tab loading:
    responses, normalized responses, and mole fractions all come from the
    stored legacy processing chain (prs_data_view / prs_corrected_response_view)
    rather than being recomputed here.
    """

    # Perseus has no run_type_num column; RUN_TYPE_MAP values are
    # hats.analysis.sample_type strings applied as a filter in load_data.
    RUN_TYPE_MAP = {
        "All": None,
        "Flasks": 'HATS',
        "PFPs": 'PFP',
        "Tanks": 'tank',
        "CCGG": 'CCGG',
    }
    # sample_type -> synthetic run_type_num so the GUI's numeric plumbing
    # (marker maps, PFP stats skip, autoscale excludes) keeps working.
    SAMPLE_TYPE_NUM = {
        'HATS': 1, 'cal': 2, 'CCGG': 3, 'test': 4, 'PFP': 5,
        'blank': 6, 'tank': 7, 'std': 8, 'burn': 9,
    }
    STANDARD_RUN_TYPE = 8       # 'std' in SAMPLE_TYPE_NUM
    STANDARD_PORT_NUM = 7       # reference tank port
    EXCLUDE = [2, 4, 6, 7, 9]   # cal/test/blank/tank/burn: keep out of autoscale
    BASE_MARKER_SIZE = 40       # smaller than the 60 default -- prs loads a
                                 # full month of points per plot, not one run
    # Every group except 'std' (8) is consolidated into one legend/plot entry
    # spanning many distinct samples/cylinders/sites -- a mean+-std across
    # them is meaningless. Only the reference tank the data is normalized to
    # (std, run_type_num 8) gets a stats line.
    STATS_RUN_TYPES = (8,)
    # S0-S5 "beginning of a new sensitivity level" tags delimit the segments
    # the reference-tank smoothing runs between; they serve as run_time here.
    S_TAG_NUMS = (271, 272, 273, 274, 275, 276)

    MARKER_MAP = {
        # synthetic run_type_num
        0: '*',   # unknown
        1: 'o',   # HATS flask
        2: '^',   # cal tank
        3: 'h',   # CCGG flask
        4: 'P',   # test
        5: 's',   # PFP
        6: 'v',   # blank
        7: '^',   # tank
        8: 'D',   # std
        9: 'v',   # burn
    }

    def __init__(self):
        super().__init__(inst_id='pr1')
        self.inst_id = 'prs'
        self.inst_num = self.INSTRUMENTS['pr1']
        self.inst_nums = (self.INSTRUMENTS['pr1'], self.INSTRUMENTS['pr2'])
        self.calibration_inst_ids = ('PR1', 'PR2')
        self.start_date = '20100101'
        self.gc_dir = Path("/hats/gc/pr1")
        self.response_type = 'response'     # column name built in load_data
        self.analytes = self.query_analytes()
        self.molecules = self.analytes.keys()
        self.analytes_inv = {v: k for k, v in self.analytes.items()}

    def query_analytes(self):
        """Use the PR1 analyte list for the combined Perseus system, ordered by
        disp_order (falling back to param_num where disp_order is unset)."""
        sql = f"""
            SELECT param_num, channel, display_name
            FROM hats.analyte_list
            WHERE inst_num = {self.INSTRUMENTS['pr1']}
            ORDER BY (disp_order IS NULL), disp_order, param_num;
        """
        df = pd.DataFrame(self.doquery(sql))
        if df.empty:
            return {}
        return dict(zip(df['display_name'], df['param_num']))

    def _segment_breaks(self, pnum, inst_num, start_date=None, end_date=None):
        """Return sorted S-tag (271-276) datetimes for one parameter on one
        instrument.

        These mark "beginning of a new sensitivity level" -- the boundaries
        the legacy reference-tank smoothing runs between -- and stand in for
        run_time, which Perseus has no native concept of. Looks back up to 60
        days before start_date (segments run up to ~42 days long) so the
        carry-in break for a segment that started before the window is still
        found, and a day past end_date for a break landing on the edge.

        PR1 (58) and PR2 (238) are different physical mass spectrometers, not
        one relabeled system -- a segment must never be computed as spanning
        both, so this always filters to a single inst_num rather than the
        pooled ``inst_nums`` tuple.
        """
        date_filter = ""
        params = [pnum, inst_num]
        if start_date is not None and end_date is not None:
            date_filter = "AND a.analysis_datetime BETWEEN %s AND %s"
            params += [
                (pd.Timestamp(start_date) - pd.Timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S'),
                (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
            ]
        tag_ph = ",".join(str(t) for t in self.S_TAG_NUMS)
        sql = f"""
            SELECT DISTINCT a.analysis_datetime
            FROM hats.flags_internal f
            JOIN hats.analysis a ON a.num = f.analysis_num
            WHERE f.parameter_num = %s
                AND a.inst_num = %s
                AND f.tag_num IN ({tag_ph})
                {date_filter}
            ORDER BY a.analysis_datetime;
        """
        rows = self.doquery(sql, params)
        if not rows:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(pd.to_datetime([r['analysis_datetime'] for r in rows], utc=True))

    def _assign_run_time(self, df, pnum, start_date=None, end_date=None):
        """Assign df['run_time'] = the S-tag segment start <= analysis_datetime.

        Rows preceding the first known break fall back to the first
        analysis_datetime present *for that instrument* in df (there is no
        earlier boundary to anchor to). PR1/PR2 rows are resolved against
        each instrument's own break list independently -- see
        _segment_breaks().
        """
        df['run_time'] = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')
        for inst_num, idx in df.groupby('inst_num').groups.items():
            breaks = self._segment_breaks(pnum, int(inst_num), start_date, end_date)
            dt = df.loc[idx, 'analysis_datetime']
            if breaks.empty:
                df.loc[idx, 'run_time'] = dt.min()
                continue
            positions = np.searchsorted(breaks.values, dt.values, side='right') - 1
            run_time = pd.Series(pd.NaT, index=idx, dtype='datetime64[ns, UTC]')
            found = positions >= 0
            run_time.loc[dt.index[found]] = breaks[positions[found]]
            if (~found).any():
                run_time.loc[dt.index[~found]] = dt.min()
            df.loc[idx, 'run_time'] = run_time
        return df

    def _resolve_load_dates(self, pnum, start_date, end_date):
        """Normalize start_date/end_date (YYMM or full datetime) to a
        concrete (start, end) datetime-string range shared by every
        load_data* variant.

        When start_date == end_date, resolves to the full S-tag segment
        [this break, next break) containing that timestamp, so a
        click-to-zoom / carry-in load pulls the whole sensitivity level
        rather than just the clicked instant. PR1/PR2 are different
        instruments, so the timestamp's own inst_num is looked up first
        rather than pooling both instruments' segment breaks together.
        """
        if end_date is None:
            end_date = datetime.today()
        elif len(end_date) == 4:
            # YYMM -> last instant of that month
            month_start = datetime.strptime(end_date, "%y%m")
            end_date = (month_start + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d 23:59:59")
        # else: expecting '%Y-%m-%d %H:%M:%S' format

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif len(start_date) == 4:
            start_date = datetime.strptime(start_date, "%y%m")
            start_date = start_date.strftime("%Y-%m-01")
        # else: expecting '%Y-%m-%d %H:%M:%S' format

        exact_segment = (str(start_date) == str(end_date))
        if exact_segment:
            target = pd.Timestamp(start_date, tz='UTC')
            target_naive = target.tz_convert(None).strftime('%Y-%m-%d %H:%M:%S')
            inst_row = self.doquery(
                "SELECT inst_num FROM hats.analysis "
                "WHERE inst_num IN (%s, %s) AND analysis_datetime = %s LIMIT 1",
                [self.inst_nums[0], self.inst_nums[1], target_naive],
            )
            target_inst_num = inst_row[0]['inst_num'] if inst_row else self.inst_nums[0]
            breaks = self._segment_breaks(
                pnum,
                target_inst_num,
                (target - pd.Timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S'),
                (target + pd.Timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S'),
            )
            pos = np.searchsorted(breaks.values, np.array([target.to_datetime64()]), side='right')[0] - 1
            if pos >= 0:
                seg_start = breaks[pos]
                seg_end = breaks[pos + 1] if pos + 1 < len(breaks) else (target + pd.Timedelta(days=1))
            else:
                seg_start = target - pd.Timedelta(hours=1)
                seg_end = target + pd.Timedelta(hours=1)
            start_date = seg_start.strftime('%Y-%m-%d %H:%M:%S')
            end_date = (seg_end - pd.Timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')

        return start_date, end_date

    def load_data(self, pnum, channel=None, run_type_num=None, start_date=None, end_date=None, verbose=True):
        """Load data from prs_data_view / prs_corrected_response_view.

        Read-only: response, normalized_resp ("ratio"), and mole_fraction all
        come from the already-computed legacy processing chain. run_type_num
        here is actually a sample_type string (see RUN_TYPE_MAP).

        Args:
            pnum (int): Parameter number to filter data.
            channel: unused (Perseus has no channel concept); accepted for
                signature compatibility with the GUI's generic call site.
            run_type_num (str, optional): sample_type filter, e.g. 'PFP'.
            start_date (str, optional): Start date; YYMM or full datetime.
                When equal to end_date, loads the full S-tag segment whose
                run_time equals that timestamp (click-to-zoom / carry-in load).
            end_date (str, optional): End date; YYMM or full datetime.
        """
        start_date, end_date = self._resolve_load_dates(pnum, start_date, end_date)

        run_type_filter = ""
        if run_type_num is not None:
            safe_type = str(run_type_num).replace("'", "''")
            run_type_filter = f"AND v.sample_type = '{safe_type}'"

        if verbose:
            print(f"Loading data from {start_date} to {end_date} for parameter {pnum}")

        # prs_data_view.rejected folds in tag 318 ("Preliminary data, not
        # ready for release"), an automated release-status marker rather than
        # a quality rejection -- it fires on huge swaths of otherwise-good
        # data. rejected_other_than_preliminary is the view's own column that
        # excludes just that tag; use it here instead.
        query = f"""
            SELECT
                v.analysis_num, v.analysis_datetime, v.inst_num, v.inst_id,
                v.sample_id, v.pair_id_num, v.site, v.sample_datetime,
                v.sample_type, v.port, v.standard_serial_num,
                v.value, v.rejected_other_than_preliminary AS rejected,
                v.suspicious, v.interp,
                c.blank_corrected_response, c.interpolated_std_response,
                c.normalized_response
            FROM hats.prs_data_view v
            LEFT JOIN hats.prs_corrected_response_view c
                ON c.analysis_num = v.analysis_num AND c.parameter_num = v.parameter_num
            WHERE v.inst_num IN ({self.inst_nums[0]}, {self.inst_nums[1]})
                AND v.parameter_num = {pnum}
                {run_type_filter}
                AND v.analysis_datetime BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY v.analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            if verbose:
                print(f"No data found for parameter {pnum} in the specified date range.")
            return pd.DataFrame()

        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'], errors='raise', utc=True)
        df['sample_datetime']   = pd.to_datetime(df['sample_datetime'], errors='coerce', utc=True)
        df['port']              = df['port'].astype(int)
        df['parameter_num']     = pnum
        df['channel']           = None
        df['detrend_method_num'] = 0

        df['response']       = df['blank_corrected_response'].astype(float)
        df['smoothed']        = df['interpolated_std_response'].astype(float)
        df['normalized_resp'] = df['normalized_response'].astype(float)
        df.drop(columns=['blank_corrected_response', 'interpolated_std_response',
                          'normalized_response'], inplace=True)

        df['value'] = df['value'].astype(float)
        df['mole_fraction'] = df['value'].mask(df['value'] <= -99)
        df.drop(columns=['value'], inplace=True)

        df['run_type_num'] = df['sample_type'].map(self.SAMPLE_TYPE_NUM).fillna(0).astype(int)
        df['port_info']    = df['sample_type']
        df['flask_port']   = pd.NA

        df['rejected']   = df['rejected'].fillna(0).astype(int)
        df['suspicious'] = df['suspicious'].fillna(0).astype(int)
        df['has_info_tag'] = df['suspicious'].astype(bool)

        df = self._assign_run_time(df, pnum, start_date=start_date, end_date=end_date)
        df = self.add_port_labels(df)

        return df.sort_values('analysis_datetime')

    def load_data_from_prs_tables(self, pnum, channel=None, run_type_num=None, start_date=None, end_date=None, verbose=True):
        """Load data from hats.prs_analysis / hats.prs_mole_fractions --
        itxbin's own independent (non-Matlab) computation, written by
        prs_batch.py. CFC-11 (pnum 29) only as of this writing; other
        parameters simply won't have prs_mole_fractions rows yet.

        Same output contract as load_data() (same columns, same run_time/
        segment semantics) so the rest of Perseus_Instrument/logos_data.py
        (add_port_labels, _assign_run_time's caller, the GUI) needs no
        changes to consume either source -- see logos_data.conf's
        [instrument.prs] data_source flag for how callers choose between
        the two.

        rows carrying nonlinearity_uncorrected=1 (currently PFP/CCGG/cal/
        burn/test) have a mole_fraction computed without the response-
        nonlinearity correction the legacy Matlab chain applies at low
        sample pressure -- known incomplete, not nulled, so the data stays
        visible with its caveat rather than silently disappearing.
        """
        start_date, end_date = self._resolve_load_dates(pnum, start_date, end_date)

        run_type_filter = ""
        if run_type_num is not None:
            safe_type = str(run_type_num).replace("'", "''")
            run_type_filter = f"AND a.sample_type = '{safe_type}'"

        if verbose:
            print(f"Loading data from {start_date} to {end_date} for parameter {pnum} (prs_tables source)")

        query = f"""
            SELECT
                a.legacy_analysis_num AS analysis_num, a.analysis_datetime, a.run_time,
                a.inst_num, a.sample_id, a.pair_id_num, a.site_num,
                a.sample_type, a.port, a.standard_serial_num,
                m.raw_response, m.blank_corrected_response, m.smoothed_response,
                m.normalized_resp, m.mole_fraction, m.nonlinearity_uncorrected,
                m.ref_tank_serial, m.ref_tank_coef0
            FROM hats.prs_analysis a
            JOIN hats.prs_mole_fractions m ON m.analysis_num = a.num AND m.parameter_num = {pnum}
            WHERE a.inst_num IN ({self.inst_nums[0]}, {self.inst_nums[1]})
                {run_type_filter}
                AND a.analysis_datetime BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY a.analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            if verbose:
                print(f"No prs_tables data found for parameter {pnum} in the specified date range.")
            return pd.DataFrame()

        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'], errors='raise', utc=True)
        df['run_time']          = pd.to_datetime(df['run_time'], errors='raise', utc=True)
        df['sample_datetime']   = pd.NaT   # not resolved yet in prs_analysis (see plan's out-of-scope note)
        df['port']              = df['port'].astype(int)
        df['parameter_num']     = pnum
        df['channel']           = None
        df['detrend_method_num'] = 0

        inst_id_map = dict(zip(self.inst_nums, self.calibration_inst_ids))
        df['inst_id'] = df['inst_num'].map(inst_id_map)
        df['site']    = None  # site_num -> site code lookup not needed for the demo; PFP/HATS labels fall back gracefully

        df['response']       = df['raw_response'].astype(float)
        df['smoothed']        = df['smoothed_response'].astype(float)
        df['normalized_resp'] = df['normalized_resp'].astype(float)
        df['mole_fraction']   = df['mole_fraction'].astype(float)
        df['nonlinearity_uncorrected'] = df['nonlinearity_uncorrected'].fillna(0).astype(bool)

        df['run_type_num'] = df['sample_type'].map(self.SAMPLE_TYPE_NUM).fillna(0).astype(int)
        df['port_info']    = df['sample_type']
        df['flask_port']   = pd.NA

        # rejected/suspicious tagging is not wired up for prs_tables yet --
        # this source is compute-only, no tag_mole_fractions write path exists.
        df['rejected']     = 0
        df['suspicious']   = 0
        df['has_info_tag'] = df['nonlinearity_uncorrected']  # surface the known-gap flag via the existing info-tag overlay

        df = self.add_port_labels(df)

        return df.sort_values('analysis_datetime')

    def add_port_labels(self, df):
        """Helper to add port labels, colors, and markers for the Perseus plot."""
        multi_inst = df['inst_id'].nunique() > 1

        def _inst_suffix(row):
            return f" [{row['inst_id']}]" if multi_inst else ""

        df['port_label'] = df['sample_type'].astype(str)

        # std / blank / burn / test / cal / tank: label by serial/sample id
        # when the loaded window only ever saw one of them (the common case
        # for std/blank/tank -- a reference cylinder installed for months) --
        # otherwise fall back to the bare category name so the legend gets
        # one stable label instead of an arbitrary cylinder's serial standing
        # in for a group that actually rotated through many.
        simple_mask = df['sample_type'].isin(['std', 'blank', 'burn', 'test', 'cal', 'tank'])
        for st, grp_idx in df.loc[simple_mask].groupby('sample_type').groups.items():
            if df.loc[grp_idx, 'sample_id'].nunique() == 1:
                df.loc[grp_idx, 'port_label'] = f"{st} {df.loc[grp_idx, 'sample_id'].iat[0]}"

        # HATS/PFP/CCGG samples share one port_idx per category (see below) --
        # a generic group label here; per-flask site/sample/pair detail is
        # still available per-point in the click tooltip via the raw columns.
        df.loc[df['sample_type'] == 'HATS', 'port_label'] = 'Flasks'
        df.loc[df['sample_type'] == 'PFP', 'port_label'] = 'PFPs'
        df.loc[df['sample_type'] == 'CCGG', 'port_label'] = 'CCGG'

        if multi_inst:
            df['port_label'] = df['port_label'] + df.apply(_inst_suffix, axis=1)

        df['port_label'] = (
            df['port_label']
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

        # assign colors to sites (HATS/PFP/CCGG)
        cmap = plt.get_cmap('tab20')
        site_colors = {site: cmap(i % 20) for i, site in enumerate(self.LOGOS_sites)}
        df['port_color'] = df['site'].map(site_colors).fillna('gray').astype(object)

        # std/cal/tank: distinct color per cylinder. sample_id is the
        # cylinder identity for all three (for 'std' it's the reference
        # standard itself; for 'cal'/'tank' it's the tank being calibrated,
        # which differs from standard_serial_num -- that column names the
        # *reference* tank used to normalize it, not the sample's own identity).
        std_like = df['sample_type'].isin(['std', 'cal', 'tank'])
        if std_like.any():
            serials = df.loc[std_like, 'sample_id'].astype(str)
            serial_codes = pd.factorize(serials)[0]
            tank_cmap = plt.get_cmap('tab10')
            colors = pd.Series(
                [tank_cmap(c % 10) for c in serial_codes], index=df.index[std_like], dtype=object
            )
            df.loc[std_like, 'port_color'] = colors
        df.loc[df['sample_type'] == 'std', 'port_color'] = 'red'
        df.loc[df['sample_type'].isin(['blank', 'burn']), 'port_color'] = 'black'
        df.loc[df['sample_type'] == 'test', 'port_color'] = 'gray'

        # legend_color: the swatch color for each port_idx group's legend
        # entry. Equal to port_color everywhere except groups that mix many
        # per-point colors (site for HATS/PFP/CCGG, serial for cal/tank) --
        # port_color.iloc[0] there would show one arbitrary member's color as
        # if it applied to the whole group, so use a fixed representative
        # color per category whenever the group has more than one identity.
        df['legend_color'] = df['port_color']
        df.loc[df['sample_type'] == 'HATS', 'legend_color'] = 'steelblue'
        df.loc[df['sample_type'] == 'PFP', 'legend_color'] = 'darkorange'
        df.loc[df['sample_type'] == 'CCGG', 'legend_color'] = 'seagreen'
        for st, fallback in (('cal', 'purple'), ('tank', 'saddlebrown')):
            mask = df['sample_type'] == st
            if mask.any() and df.loc[mask, 'sample_id'].nunique() > 1:
                df.loc[mask, 'legend_color'] = fallback

        # port_idx: one group per sample_type (run_type_num already gives
        # this). Perseus loads a whole month of continuous flask/PFP traffic
        # at once -- factorizing every individual flask/package (as M4 does
        # for its much smaller per-run PFP counts) produced 500+ scatter
        # artists and legend rows in a typical month and made the plot very
        # slow to render. Site identity is kept as the per-point scatter
        # color (see add_port_labels callers); per-flask detail lives in the
        # point tooltip instead of the legend.
        df['port_idx'] = df['run_type_num'].astype('Int64')

        df['port_marker'] = df['run_type_num'].map(self.MARKER_MAP).fillna('o')

        return df


PRS_Instrument = Perseus_Instrument
