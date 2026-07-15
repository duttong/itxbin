"""In-situ GC instrument classes: IE3, CATS (per-site), and BLD1.
Import public names through the logos_instruments facade."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from datetime import datetime, timedelta
import time

from logos_instruments_core import HATS_DB_Functions, Normalizing

class IE3_Instrument(HATS_DB_Functions):
    """Class for accessing IE3 specific functions in the HATS database.
       Currently only one deployed at SMO, but code is structured to support multiple sites if needed in the future.
    """

    RUN_TYPE_MAP = {
        "All": None,
        "Air Samples": "air",
        "Calibrations": "cal",
    }
    DEFAULT_ANALYTE_NAME = "N2O (a)"
    DEFAULT_ANALYTE_CHANNEL = "a"
    STANDARD_PORT_NUM = 5
    CAL1_PORT = 9
    CAL2_PORT = 1
    AIR_PORTS = [3, 7]
    EXCLUDE = [1, 2, 5, 9]  # tank and stop ports; autoscale samples uses only air ports 3 & 7
    AUTOSCALE_STANDARD_PORTS = [1, 5, 9]  # ref tank + high/low standards

    # IE3 ran pre-production test data before 2026; hide it from the GUI run
    # list and timeseries. CATS (subclass) overrides this to None to keep its
    # full record. load_data is intentionally not floored so batch/programmatic
    # callers can still reach any date.
    DATA_START_DATE = '2026-01-01'

    # mf_method_num values in hats.ng_insitu_mole_fractions; the calibration
    # method is recorded per-week per-analyte on the air rows themselves.
    MF_METHOD_REF = 1            # mf = normalized_resp * coef0(ref tank); no weekly fit
    MF_METHOD_SCALE_SIMPLE = 1   # alias for MF_METHOD_REF (scale-simple to ref tank)
    MF_METHOD_CAL12 = 2          # 2-point weekly fit through both cal tanks
    MF_METHOD_CAL1 = 3           # single tank through origin, CAL1_PORT (9)
    MF_METHOD_CAL2 = 4           # single tank through origin, CAL2_PORT (1)
    MF_METHOD_LABELS = {1: 'ref', 2: 'cal12', 3: 'cal1', 4: 'cal2'}
    # methods 2/3/4 build a weekly ng_response fit; method 1 (ref) does not.
    NG_RESPONSE_METHODS = (2, 3, 4)

    def __init__(self, site: str = "smo"):
        super().__init__('ie3')
        site = site.lower()
        valid_sites_dict = self.get_valid_sites()
        if site not in valid_sites_dict:
            raise ValueError(f"Invalid site {site!r}. Valid sites: {sorted(valid_sites_dict.keys())}")

        self.site = site
        self.site_num = valid_sites_dict[site]
        self.start_date = '20251001'
        self.gc_dir = Path(f"/hats/gc/{site}")
        self.export_dir = self.gc_dir / "results"
        self.response_type = 'height'

        # query raw molecule/analyte dicts
        self.molecules = self.query_molecules()
        analyte_rows = self.db.doquery(
            "SELECT display_name, param_num, channel "
            f"FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
        )
        df_analytes = pd.DataFrame(analyte_rows)
        df_analytes['channel'] = df_analytes['channel'].fillna('').astype(str).str.lower().str.strip()
        df_analytes = df_analytes.sort_values(['channel', 'display_name'])
        df_analytes['display_name_ch'] = df_analytes.apply(
            lambda r: f"{r['display_name']} ({r['channel']})" if r['channel'] else r['display_name'],
            axis=1,
        )
        self.analytes = dict(zip(df_analytes['display_name_ch'], df_analytes['param_num']))
        self.analytes_inv = {
            int(v): k for k, v in self.analytes.items()
        }
        self.analytes_inv[None] = self.DEFAULT_ANALYTE_NAME
        self.port_config = self._load_port_config()
        self.norm = Normalizing(self.inst_id, self.STANDARD_PORT_NUM, 'port', self.response_type)

    def get_valid_sites(self):
        """ Returns a dictionary of valid IE3 site codes and site numbers. """
        sql = "SELECT num, code FROM gmd.site WHERE code IN ('SMO', 'BRW', 'SPO', 'MLO');"
        df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            return {}
        return dict(zip(df['code'].str.lower(), df['num']))

    def _load_port_config(self):
        """ Load port configuration once from the database. 
            If there was a tank change, in the period of data processing, this 
            will not capture that. But for simplicity, we assume the port configuration is stable over time.
            TODO: if needed, we could modify this to return a time-varying configuration.
        """
        sql = """
            SELECT start_datetime, site_num, port_num, abbr, serial_number FROM ng_port_info pi 
            JOIN ng_port_inlet_types pt ON pi.port_type_num=pt.num;
        """
        df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            return None
            
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce', utc=True)
        df = df.sort_values('start_datetime')
        df['label'] = df['serial_number'].fillna(df['abbr'])
        config = df.drop_duplicates(subset=['site_num', 'port_num'], keep='last')
        return config[['site_num', 'port_num', 'label']]

    def query_return_run_list(self, runtype=None, start_date=None, end_date=None):
        """Return run_time list for IE3 from ng_insitu_analysis."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

        # Hide the pre-production test window (IE3 only; None for CATS).
        floor = f"AND run_time >= '{self.DATA_START_DATE}'" if self.DATA_START_DATE else ""

        sql = f"""
            SELECT DISTINCT run_time
            FROM hats.ng_insitu_analysis
            WHERE inst_num = {self.inst_num}
                AND site_num = (SELECT num FROM gmd.site WHERE lower(code) = '{self.site}')
                AND run_time BETWEEN '{start_date}' AND '{end_date}'
                {floor}
            ORDER BY run_time;
        """
        df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            return []
        df['run_time'] = pd.to_datetime(df['run_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df['run_time'].to_list()

    def load_data(
        self,
        pnum,
        channel=None,
        run_type_num=None,
        start_date=None,
        end_date=None,
        site_num=None,
        verbose=True,
    ):
        """Load data from ng_insitu tables with date filtering.

        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
            site_num (int, optional): Site number filter. Defaults to None.
        """
        if pnum is None:
            pnum = self._default_pnum()
            if pnum is None:
                if verbose:
                    print(
                        "IE3 load_data called with pnum=None and no default analyte found; "
                        "returning empty DataFrame."
                    )
                return pd.DataFrame()
        if isinstance(start_date, str) and " (" in start_date:
            start_date = start_date.split(" (")[0]
        if isinstance(end_date, str) and " (" in end_date:
            end_date = end_date.split(" (")[0]

        if end_date is None:
            end_date = datetime.today()
        elif len(end_date) == 4:
            end_date = datetime.strptime(end_date, "%y%m")
            end_date = end_date.strftime("%Y-%m-31")

        if start_date is None:
            start_date = end_date - timedelta(days=30)
        elif len(start_date) == 4:
            start_date = datetime.strptime(start_date, "%y%m")
            start_date = start_date.strftime("%Y-%m-01")

        channel_str = f"AND channel = '{channel}'" if channel else ""
        if site_num is None:
            site_num = self.site_num
        site_str = f"AND site_num = {site_num}" if site_num is not None else ""

        if verbose:
            print(f"Loading {self.inst_id.upper()} data from {start_date} to {end_date} for parameter {pnum}")

        if str(start_date) == str(end_date):
            time_filter = f"AND run_time = '{start_date}'"
        else:
            time_filter = f"AND analysis_time BETWEEN '{start_date}' AND '{end_date}'"

        query = f"""
            SELECT
                analysis_num,
                mf_num,
                analysis_time,
                run_time,
                site_num,
                inst_num,
                port,
                parameter_num,
                channel,
                ng_response_id,
                height,
                area,
                retention_time,
                mole_fraction,
                unc,
                rejected,
                sample_loop_temp,
                sample_loop_pressure,
                sample_loop_flow,
                detrend_method_num,
                mf_method_num
            FROM hats.ng_insitu_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                {channel_str}
                {site_str}
                AND (height IS NULL OR height <> -999)
                {time_filter}
            ORDER BY analysis_time;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            if verbose:
                print(f"No data found for parameter {pnum} in the specified date range.")
            return pd.DataFrame()

        df['analysis_time'] = pd.to_datetime(df['analysis_time'], errors='raise', utc=True)
        df['analysis_datetime'] = df['analysis_time']
        df['run_time'] = pd.to_datetime(df['run_time'], errors='raise', utc=True)
        df['parameter_num'] = df['parameter_num'].astype(int)
        df['run_type_num'] = 0  # IE3 doesn't have run types, so set to 0 or some default
        df['height'] = df['height'].astype(float)
        df['area'] = df['area'].astype(float)
        df['retention_time'] = df['retention_time'].astype(float)

        df['rejected'] = df['rejected'].fillna(0).astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].fillna(5).astype(int)
        df['mf_method_num'] = df['mf_method_num'].fillna(self.default_mf_method(pnum)).astype(int)
        df = self.norm.merge_smoothed_data(df)
        df = self.add_port_labels(df)

        return df.sort_values('analysis_datetime')

    def _default_pnum(self) -> int | None:
        """Return a default parameter number for IE3."""
        if self.DEFAULT_ANALYTE_NAME in self.analytes:
            return self.analytes[self.DEFAULT_ANALYTE_NAME]
        sql = (
            "SELECT display_name, param_num "
            "FROM hats.analyte_list "
            f"WHERE inst_num = {self.inst_num} "
            f"AND channel = '{self.DEFAULT_ANALYTE_CHANNEL}' "
            "AND display_name LIKE 'CFC11%' "
            "ORDER BY display_name "
            "LIMIT 1;"
        )
        rows = self.db.doquery(sql)
        if rows:
            return rows[0]["param_num"]
        return None

    def return_preferred_channel(self) -> pd.DataFrame:
        """Return preferred channel assignments from hats.ng_preferred_channel for this instrument."""
        sql = """
        SELECT inst_num, parameter_num, start_date, channel
        FROM hats.ng_preferred_channel
        WHERE inst_num = %s
        ORDER BY parameter_num, start_date
        """
        return pd.DataFrame(self.db.doquery(sql, (self.inst_num,)))

    def upsert_calibrations(self, df, parameter_num):
        """IE3 does not write to hats.calibrations."""
        return

    def upsert_mole_fractions(self, df, response_id=None):
        """
        Update mole_fraction, unc, ng_response_id, detrend_method_num, and
        mf_method_num in hats.ng_insitu_mole_fractions.
        Overrides the base class which writes to ng_mole_fractions (wrong table for IE3).
        """
        if df.empty or 'mole_fraction' not in df.columns:
            return
        sql = """
            UPDATE hats.ng_insitu_mole_fractions
            SET mole_fraction = %s, unc = %s, ng_response_id = %s,
                detrend_method_num = %s, mf_method_num = %s
            WHERE analysis_num = %s
              AND parameter_num = %s
              AND channel = %s;
        """
        df = df.copy()
        df['mole_fraction'] = (
            pd.to_numeric(df['mole_fraction'], errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .round(5)
        )
        if 'mf_method_num' not in df.columns:
            df['mf_method_num'] = df['parameter_num'].map(self.default_mf_method)
        else:
            missing = df['mf_method_num'].isna()
            if missing.any():
                df.loc[missing, 'mf_method_num'] = df.loc[missing, 'parameter_num'].map(self.default_mf_method)
        if 'unc' not in df.columns:
            df['unc'] = np.nan
        if 'ng_response_id' not in df.columns:
            df['ng_response_id'] = None
        for _, row in df.iterrows():
            mf = row['mole_fraction']
            unc = row['unc']
            rid = row['ng_response_id']
            self.db.doquery(sql, [
                None if pd.isna(mf) else float(mf),
                None if pd.isna(unc) else float(unc),
                None if pd.isna(rid) else int(rid),
                int(row['detrend_method_num']),
                int(row['mf_method_num']),
                row['analysis_num'],
                row['parameter_num'],
                row['channel'],
            ])

    def update_flags_all_analytes(self, df):
        """
        Propagate newly applied tags from df to all parameter rows sharing the same analysis_num.
        Overrides the base class which joins on ng_mole_fractions/ng_analysis (wrong tables for IE3).
        """
        if df.empty or '_pending_tag_num' not in df.columns:
            return

        tagged = df.loc[df['_pending_tag_num'].notna(), ['analysis_num', '_pending_tag_num']]
        if tagged.empty:
            return

        for tag_num, group in tagged.groupby('_pending_tag_num'):
            analysis_nums = group['analysis_num'].dropna().astype(int).unique().tolist()
            if not analysis_nums:
                continue
            placeholders = ','.join(['%s'] * len(analysis_nums))
            sql_set = f"""
                INSERT IGNORE INTO hats.ng_insitu_mole_fraction_tags (
                    ng_insitu_mole_fraction_num,
                    tag_num
                )
                SELECT num, %s
                FROM hats.ng_insitu_mole_fractions
                WHERE analysis_num IN ({placeholders});
            """
            self.db.doquery(sql_set, [int(tag_num), *analysis_nums])

    def calc_mole_fraction_scale_simple(self, df):
        """Compute mole_fraction = normalized_resp * coef0 from hats.scale_assignments.

        Uses the tank on STANDARD_PORT_NUM (port 5) from port_config and looks up
        coef0 for each parameter_num. This is a temporary calculation path until
        a proper IE3 response calibration is in place.
        """
        if df.empty:
            df_out = df.copy()
            df_out['mole_fraction'] = pd.Series(dtype='float64')
            return df_out

        pnum = int(df['parameter_num'].iat[0])

        # Resolve the standard tank serial number from port_config
        ref_tank = None
        if self.port_config is not None:
            mask = (
                (self.port_config['site_num'] == self.site_num)
                & (self.port_config['port_num'] == self.STANDARD_PORT_NUM)
            )
            rows = self.port_config.loc[mask, 'label']
            if not rows.empty:
                ref_tank = rows.iat[0]

        if ref_tank is None:
            out = df.copy()
            out['mole_fraction'] = np.nan
            return out

        # Pick the ref-tank fill in use over this batch of data. These rows are
        # a single processing window, so a representative date is sufficient.
        run_date = None
        for col in ('analysis_datetime', 'analysis_time', 'run_time'):
            if col in df.columns and df[col].notna().any():
                run_date = pd.to_datetime(df[col]).max()
                break

        coefs = self.scale_assignments(ref_tank, pnum, run_date=run_date)
        if coefs is None:
            out = df.copy()
            out['mole_fraction'] = np.nan
            return out

        coef0 = float(coefs['coef0'])
        out = df.copy()
        out['mole_fraction'] = out['normalized_resp'] * coef0
        return out

    def calc_mole_fraction(self, df):
        """Route by mf_method_num: method 1 → scale_simple; 2/3/4 → cal_fit.

        Rows with a NULL mf_method_num (never recorded) default per-row to
        default_mf_method(parameter_num) -- cal12 for most analytes, not a
        flat 1/ref -- so an unset method doesn't silently mean "ref".
        """
        if df.empty:
            out = df.copy()
            out['mole_fraction'] = pd.Series(dtype='float64')
            return out

        df = df.copy()
        if 'mf_method_num' not in df.columns:
            df['mf_method_num'] = np.nan
        missing = df['mf_method_num'].isna()
        if missing.any():
            df.loc[missing, 'mf_method_num'] = df.loc[missing, 'parameter_num'].map(self.default_mf_method)
        df['mf_method_num'] = df['mf_method_num'].astype(int)

        method_nums = df['mf_method_num'].unique()
        if len(method_nums) == 1 and method_nums[0] == 1:
            return self.calc_mole_fraction_scale_simple(df)

        parts = []
        for method, grp in df.groupby('mf_method_num'):
            if method == 1:
                parts.append(self.calc_mole_fraction_scale_simple(grp))
            else:
                parts.append(self.calc_mole_fraction_cal_fit(grp))
        return pd.concat(parts).sort_values('analysis_datetime')

    def calc_mole_fraction_cal_fit(self, df):
        """Compute mole_fraction from a stored ng_response weekly cal fit.

        For each row, finds the most recent ng_response row with
        run_date <= analysis_date for (inst_num, site, channel, scale_num).

        method 2 (cal12): mf = coef1 * normalized_resp + coef0
        method 3 (cal1):  mf = coef1 * normalized_resp
        method 4 (cal2):  mf = coef1 * normalized_resp
        """
        if df.empty:
            out = df.copy()
            out['mole_fraction'] = pd.Series(dtype='float64')
            out['unc'] = pd.Series(dtype='float64')
            out['ng_response_id'] = pd.Series(dtype='object')
            return out

        pnum = int(df['parameter_num'].iat[0])
        channel = str(df['channel'].iat[0])

        # Fetch the current scale_num for this parameter
        scale_rows = self.db.doquery(
            f"SELECT idx FROM reftank.scales WHERE parameter_num={pnum} AND current=1"
        )
        if not scale_rows:
            out = df.copy()
            out['mole_fraction'] = np.nan
            out['unc'] = np.nan
            out['ng_response_id'] = None
            return out
        scale_num = int(scale_rows[0]['idx'])

        # Load all ng_response fits for this inst/site/channel/scale
        fits_rows = self.db.doquery(f"""
            SELECT id, run_date, coef0, coef1, unc_fit, sigma_ref
            FROM hats.ng_response
            WHERE inst_num = {self.inst_num}
              AND site = '{self.site}'
              AND channel = '{channel}'
              AND scale_num = {scale_num}
            ORDER BY run_date
        """)
        if not fits_rows:
            out = df.copy()
            out['mole_fraction'] = np.nan
            out['unc'] = np.nan
            out['ng_response_id'] = None
            return out

        fits = pd.DataFrame(fits_rows)
        fits['run_date'] = pd.to_datetime(fits['run_date'], utc=True)

        out = df.copy()
        mf = pd.Series(index=df.index, dtype=float)
        unc = pd.Series(index=df.index, dtype=float)
        rid = pd.Series(index=df.index, dtype=object)

        for idx, row in df.iterrows():
            analysis_dt = pd.to_datetime(row['analysis_datetime'], utc=True)
            mask = fits['run_date'] <= analysis_dt
            if not mask.any():
                continue
            fit = fits.loc[mask].iloc[-1]
            method = int(row['mf_method_num']) if pd.notna(row.get('mf_method_num')) else 2
            resp = row['normalized_resp']
            if pd.isna(resp):
                continue
            if method == 2:
                mf.at[idx] = float(fit['coef1']) * resp + float(fit['coef0'])
            else:
                mf.at[idx] = float(fit['coef1']) * resp
            unc.at[idx] = np.sqrt(
                float(fit['unc_fit']) ** 2
                + (float(fit['coef1']) * float(fit['sigma_ref'])) ** 2
            )
            rid.at[idx] = int(fit['id'])

        out['mole_fraction'] = mf
        out['unc'] = unc
        out['ng_response_id'] = rid
        return out

    def nearby_response_fits(self, pnum, channel, week_start, n=5):
        """Return the n stored hats.ng_response weekly fits nearest in time to
        week_start (excluding week_start itself) for this analyte/channel.

        Used by the calibration view's "Other Curves" overlay to compare fit
        stability across nearby weeks. Only meaningful for methods that store
        an ng_response fit (cal12/cal1/cal2); returns an empty frame if no
        scale or no other fits are found.
        """
        cols = ['run_date', 'coef0', 'coef1']
        scale_rows = self.db.doquery(
            f"SELECT idx FROM reftank.scales WHERE parameter_num={pnum} AND current=1"
        )
        if not scale_rows:
            return pd.DataFrame(columns=cols)
        scale_num = int(scale_rows[0]['idx'])

        rows = self.db.doquery(f"""
            SELECT run_date, coef0, coef1
            FROM hats.ng_response
            WHERE inst_num = {self.inst_num}
              AND site = '{self.site}'
              AND channel = '{channel}'
              AND scale_num = {scale_num}
            ORDER BY run_date
        """)
        if not rows:
            return pd.DataFrame(columns=cols)

        fits = pd.DataFrame(rows)
        fits['run_date'] = pd.to_datetime(fits['run_date'])
        target = pd.Timestamp(week_start)
        fits = fits.loc[fits['run_date'] != target].copy()
        if fits.empty:
            return fits
        fits['_dt'] = (fits['run_date'] - target).abs()
        return fits.sort_values('_dt').head(n).drop(columns='_dt')

    def upsert_ng_response(self, inst_num, site, run_date, channel, scale_num,
                           coef0, coef1, unc_fit, sigma_ref, serial_number):
        """Insert or update a weekly cal fit row in hats.ng_response.

        Uniqueness key: (inst_num, site, run_date, channel, scale_num).
        Returns the row id.
        """
        run_date_str = pd.Timestamp(run_date).strftime('%Y-%m-%d %H:%M:%S')
        existing = self.db.doquery(f"""
            SELECT id FROM hats.ng_response
            WHERE inst_num = {inst_num}
              AND site = '{site}'
              AND run_date = '{run_date_str}'
              AND channel = '{channel}'
              AND scale_num = {scale_num}
        """)
        if existing:
            row_id = existing[0]['id']
            self.db.doquery(f"""
                UPDATE hats.ng_response
                SET coef0={coef0}, coef1={coef1}, coef2=0, coef3=0,
                    unc_fit={unc_fit}, sigma_ref={sigma_ref},
                    serial_number='{serial_number}', function='poly', flag='.'
                WHERE id={row_id}
            """)
        else:
            self.db.doquery(f"""
                INSERT INTO hats.ng_response
                    (inst_num, site, run_date, channel, scale_num,
                     coef0, coef1, coef2, coef3, unc_fit, sigma_ref,
                     serial_number, function, flag)
                VALUES
                    ({inst_num}, '{site}', '{run_date_str}', '{channel}', {scale_num},
                     {coef0}, {coef1}, 0, 0, {unc_fit}, {sigma_ref},
                     '{serial_number}', 'poly', '.')
            """)
            row = self.db.doquery(f"""
                SELECT id FROM hats.ng_response
                WHERE inst_num = {inst_num}
                  AND site = '{site}'
                  AND run_date = '{run_date_str}'
                  AND channel = '{channel}'
                  AND scale_num = {scale_num}
            """)
            row_id = row[0]['id'] if row else None
        return row_id

    def query_cal_run_dates(self, pnum, channel, start_date=None, end_date=None):
        """Return list of weekly cal fit dates from ng_response for this site/channel.

        Used by the logos_data run selector to add '(Cal)' entries.
        Returns list of 'YYYY-MM-DD HH:MM:SS' strings.
        """
        scale_rows = self.db.doquery(
            f"SELECT idx FROM reftank.scales WHERE parameter_num={pnum} AND current=1"
        )
        if not scale_rows:
            return []
        scale_num = int(scale_rows[0]['idx'])

        channel_str = channel or ''
        date_filter = ""
        if start_date:
            date_filter += f" AND run_date >= '{start_date}'"
        if end_date:
            date_filter += f" AND run_date <= '{end_date}'"

        rows = self.db.doquery(f"""
            SELECT run_date FROM hats.ng_response
            WHERE inst_num = {self.inst_num}
              AND site = '{self.site}'
              AND channel = '{channel_str}'
              AND scale_num = {scale_num}
              {date_filter}
            ORDER BY run_date
        """)
        if not rows:
            return []
        return [
            pd.Timestamp(r['run_date']).strftime('%Y-%m-%d %H:%M:%S')
            for r in rows
        ]

    def default_mf_method(self, pnum):
        """Default calibration method for an analyte with no recorded value.

        CCl4 (pnum 37) defaults to cal1 (single tank through origin); every
        other analyte defaults to cal12 (2-point fit).
        """
        return self.MF_METHOD_CAL1 if int(pnum) == 37 else self.MF_METHOD_CAL12

    def uses_ng_response_fit(self, method_num):
        """True if the method builds a stored weekly ng_response fit (2/3/4).
        Method 1 (ref / scale-simple) computes mole fractions directly from the
        reference tank coef0 and stores no fit."""
        return int(method_num) in self.NG_RESPONSE_METHODS

    def fit_params_for_method(self, method_num):
        """Map an mf_method_num to (force_zero, single_port) for a weekly fit.

        cal12 -> 2-point fit through both cal tanks (force_zero=False, no single
        port); cal1 -> through origin on CAL1_PORT (9); cal2 -> through origin on
        CAL2_PORT (1).
        """
        m = int(method_num)
        if m == self.MF_METHOD_CAL1:
            return True, self.CAL1_PORT
        if m == self.MF_METHOD_CAL2:
            return True, self.CAL2_PORT
        return False, None

    def ref_tank_serial(self):
        """Serial number of the reference tank on STANDARD_PORT_NUM (port 5)."""
        if self.port_config is None:
            return None
        mask = ((self.port_config['site_num'] == self.site_num)
                & (self.port_config['port_num'] == self.STANDARD_PORT_NUM))
        rows = self.port_config.loc[mask, 'label']
        return rows.iat[0] if not rows.empty else None

    def ref_tank_coef0(self, pnum):
        """Reference-tank coef0 (assigned value) for this parameter, or None.
        This is the slope of the method-1 (ref) scaling: mf = coef0 * resp."""
        serial = self.ref_tank_serial()
        if serial is None:
            return None
        coefs = self.scale_assignments(serial, pnum)
        if not coefs or coefs.get('coef0') is None:
            return None
        return float(coefs['coef0'])

    def ref_tank_unc_c0(self, pnum):
        """Reference-tank unc_c0 (uncertainty of the assigned coef0) for this
        parameter, or None."""
        serial = self.ref_tank_serial()
        if serial is None:
            return None
        coefs = self.scale_assignments(serial, pnum)
        if not coefs or coefs.get('unc_c0') is None:
            return None
        return float(coefs['unc_c0'])

    def _week_air_filter(self, pnum, channel, week_start):
        """Return SQL fragments (where, t0, t1) selecting this analyte's air
        rows within the week [week_start, week_start+7d)."""
        t0 = pd.Timestamp(week_start).strftime('%Y-%m-%d')
        t1 = (pd.Timestamp(week_start) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        air_in = ', '.join(str(p) for p in self.AIR_PORTS)
        channel_str = f"AND mf.channel = '{channel}'" if channel else ""
        where = (
            f"a.inst_num = {self.inst_num} "
            f"AND a.site_num = {self.site_num} "
            f"AND a.port IN ({air_in}) "
            f"AND mf.parameter_num = {pnum} "
            f"{channel_str} "
            f"AND a.analysis_time >= '{t0}' AND a.analysis_time < '{t1}'"
        )
        return where

    def get_week_mf_method(self, pnum, channel, week_start):
        """Return the recorded cal method (cal12/cal1/cal2) for this
        analyte/channel's air rows in the given week.

        Any value that is not a valid IE3 cal method (e.g. a legacy
        scale-simple 1, or NULL) is coerced to default_mf_method() so the GUI
        and batch always agree on a concrete cal method.
        """
        where = self._week_air_filter(pnum, channel, week_start)
        rows = self.db.doquery(f"""
            SELECT mf.mf_method_num AS m, COUNT(*) AS n
            FROM hats.ng_insitu_mole_fractions mf
            JOIN hats.ng_insitu_analysis a ON a.num = mf.analysis_num
            WHERE {where}
              AND mf.mf_method_num IS NOT NULL
            GROUP BY mf.mf_method_num
            ORDER BY n DESC
            LIMIT 1
        """)
        if rows:
            m = int(rows[0]['m'])
            if m in self.MF_METHOD_LABELS:
                return m
        return self.default_mf_method(pnum)

    def set_week_mf_method(self, pnum, channel, week_start, method_num):
        """Record the calibration method for this analyte/channel by writing
        mf_method_num on the air rows in the given week only. Returns nothing.
        """
        where = self._week_air_filter(pnum, channel, week_start)
        self.db.doquery(f"""
            UPDATE hats.ng_insitu_mole_fractions mf
            JOIN hats.ng_insitu_analysis a ON a.num = mf.analysis_num
            SET mf.mf_method_num = {int(method_num)}
            WHERE {where}
        """)

    def load_cal_week_data(self, pnum, channel, week_start):
        """Load cal+ref port data for the week starting on week_start.

        Returns same DataFrame structure as load_data() but limited to
        cal ports (1, 9) and ref port (5), covering [week_start, week_start+7d).
        """
        from datetime import timedelta as _td
        t0 = pd.Timestamp(week_start).strftime('%Y-%m-%d')
        t1 = (pd.Timestamp(week_start) + _td(days=7)).strftime('%Y-%m-%d')

        cal_ports = (self.CAL1_PORT, self.CAL2_PORT, self.STANDARD_PORT_NUM)
        port_in = ', '.join(str(p) for p in cal_ports)

        channel_str = f"AND channel = '{channel}'" if channel else ""

        query = f"""
            SELECT
                analysis_num,
                mf_num,
                analysis_time,
                run_time,
                site_num,
                inst_num,
                port,
                parameter_num,
                channel,
                ng_response_id,
                height,
                area,
                retention_time,
                mole_fraction,
                unc,
                rejected,
                sample_loop_temp,
                sample_loop_pressure,
                sample_loop_flow,
                detrend_method_num,
                mf_method_num
            FROM hats.ng_insitu_data_view
            WHERE inst_num = {self.inst_num}
              AND parameter_num = {pnum}
              {channel_str}
              AND site_num = {self.site_num}
              AND port IN ({port_in})
              AND (height IS NULL OR height <> -999)
              AND analysis_time BETWEEN '{t0}' AND '{t1}'
            ORDER BY analysis_time;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            return pd.DataFrame()

        df['analysis_time'] = pd.to_datetime(df['analysis_time'], errors='raise', utc=True)
        df['analysis_datetime'] = df['analysis_time']
        df['run_time'] = pd.to_datetime(df['run_time'], errors='raise', utc=True)
        df['parameter_num'] = df['parameter_num'].astype(int)
        df['run_type_num'] = 0
        df['height'] = df['height'].astype(float)
        df['area'] = df['area'].astype(float)
        df['retention_time'] = df['retention_time'].astype(float)
        df['rejected'] = df['rejected'].fillna(0).astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].fillna(5).astype(int)
        df['mf_method_num'] = df['mf_method_num'].fillna(self.default_mf_method(pnum)).astype(int)
        df = self.norm.merge_smoothed_data(df)
        df = self.add_port_labels(df)
        return df.sort_values('analysis_datetime')

    def add_port_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple port labels and colors based on port number."""
        
        if self.port_config is not None:
            # Merge into main dataframe
            df = df.merge(
                self.port_config,
                left_on=['site_num', 'port'],
                right_on=['site_num', 'port_num'],
                how='left'
            )
            
            # Format port_label
            df['port_label'] = df.apply(
                lambda row: f"{row['label']} ({int(row['port'])})" if pd.notna(row['label']) else str(int(row['port'])),
                axis=1
            )
            df = df.drop(columns=['port_num', 'label'])
        else:
            df['port_label'] = df['port'].astype(int).astype(str)

        df['port_idx'] = df['port'].astype(int)
        df['port_marker'] = 'o'
        df['port_info'] = df['port_label']  # for compatibility with normalization and tooltips

        # Assign colors to ports using a colormap
        cmap = plt.get_cmap('tab20')
        ports = sorted(df['port'].dropna().unique())
        port_colors = {p: cmap(i % 20) for i, p in enumerate(ports)}
        df['port_color'] = df['port'].map(port_colors).fillna('gray')

        return df


class CATS_Instrument(IE3_Instrument):
    """CATS in-situ GC instrument.

    CATS has one instrument per site, each with its own inst_num (239-244).
    Data lives in the same ng_insitu_* tables as IE3.  Port layout:
      port 2 = cal1 (Std),  port 4 = air1,  port 6 = cal2 (Ref),  port 8 = air2.
    Mole fractions are computed via scale_simple: mf = normalized_resp * coef0.
    """

    INST_NUM_BY_SITE: dict[str, int] = {
        'brw': 239,
        'sum': 240,
        'nwr': 241,
        'mlo': 242,
        'smo': 243,
        'spo': 244,
    }

    RUN_TYPE_MAP = {"All": None}
    DEFAULT_ANALYTE_NAME = "N2O"
    DEFAULT_ANALYTE_CHANNEL = "q"

    # cal2 (Ref tank, port 6) is both the normalization reference and CAL2.
    STANDARD_PORT_NUM = 6
    CAL1_PORT = 2
    CAL2_PORT = 6
    AIR_PORTS = [4, 8]
    # Exclude cal tank ports from air-only autoscale; keep air ports 4 & 8.
    EXCLUDE = [2, 6]
    AUTOSCALE_STANDARD_PORTS = [2, 6]
    # CATS keeps its full multi-decade record (no pre-production test window).
    DATA_START_DATE = None

    def __init__(self, site: str = "brw"):
        site = site.lower()
        if site not in self.INST_NUM_BY_SITE:
            raise ValueError(
                f"Invalid CATS site {site!r}. Valid: {sorted(self.INST_NUM_BY_SITE)}"
            )

        # Bootstrap HATS_DB_Functions (sets up DB connection, inst_id, inst_num=239).
        # Then immediately override inst_num for the actual site.
        HATS_DB_Functions.__init__(self, 'cats')
        self.inst_num = self.INST_NUM_BY_SITE[site]

        self.site = site
        self.site_num = self._site_num_for(site)
        self.start_date = '19980101'
        self.gc_dir = Path(f"/hats/gc/{site}")
        self.export_dir = Path("/hats/gc/cats_results")
        self.response_type = 'height'

        self.molecules = self.query_molecules()
        analyte_rows = self.db.doquery(
            "SELECT display_name, param_num, channel, disp_order "
            f"FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
        ) or []
        df_analytes = pd.DataFrame(analyte_rows)
        if not df_analytes.empty:
            df_analytes['channel'] = (
                df_analytes['channel'].fillna('').astype(str).str.lower().str.strip()
            )
            df_analytes['disp_order'] = pd.to_numeric(df_analytes['disp_order'], errors='coerce')
            df_analytes = df_analytes.sort_values(
                ['disp_order', 'channel', 'display_name'], na_position='last'
            )
            df_analytes['display_name_ch'] = df_analytes.apply(
                lambda r: f"{r['display_name']} ({r['channel']})" if r['channel'] else r['display_name'],
                axis=1,
            )
            self.analytes = dict(zip(df_analytes['display_name_ch'], df_analytes['param_num']))
        else:
            self.analytes = {}

        # Add a "(pref)" entry for compounds that have ng_preferred_channel entries.
        # Inserted immediately after the last channel variant of each compound.
        pref_rows = self.db.doquery(
            "SELECT DISTINCT parameter_num FROM hats.ng_preferred_channel "
            f"WHERE inst_num = {self.inst_num};"
        ) or []
        pref_params = {int(r['parameter_num']) for r in pref_rows}
        if pref_params:
            augmented = {}
            seen_pref = set()
            # Track the last position of each compound name so we can insert after it.
            for key, pnum in self.analytes.items():
                augmented[key] = pnum
                base = key.split(" (")[0]
                pref_key = f"{base} (pref)"
                if int(pnum) in pref_params and pref_key not in seen_pref:
                    # Peek ahead: only insert after the LAST channel variant.
                    remaining_bases = [k.split(" (")[0] for k in list(self.analytes)[
                        list(self.analytes).index(key) + 1:
                    ]]
                    if base not in remaining_bases:
                        augmented[pref_key] = pnum
                        seen_pref.add(pref_key)
            self.analytes = augmented

        self.analytes_inv = {int(v): k for k, v in self.analytes.items()}
        self.analytes_inv[None] = self.DEFAULT_ANALYTE_NAME

        self.port_config = self._load_port_config()
        self.norm = Normalizing(self.inst_id, self.STANDARD_PORT_NUM, 'port', self.response_type)

    def _site_num_for(self, site: str) -> int:
        rows = self.db.doquery(
            f"SELECT num FROM gmd.site WHERE lower(code) = '{site}';"
        )
        return int(rows[0]['num'])

    def get_valid_sites(self) -> dict:
        codes = ",".join(f"'{s.upper()}'" for s in self.INST_NUM_BY_SITE)
        sql = f"SELECT num, code FROM gmd.site WHERE code IN ({codes});"
        df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            return {}
        return dict(zip(df['code'].str.lower(), df['num']))

    def calc_mole_fraction(self, df):
        return self.calc_mole_fraction_scale_simple(df)


class BLD1_Instrument(HATS_DB_Functions):
    """ Class for accessing BLD1 (Stratcore) specific functions in the HATS database. """
    
    RUN_TYPE_MAP = {
        # Name: run_type_num
        "All": None,        # no filter
        "Calibrations": 2,
        "Other": 4,
        "Zero": 6,
        "Test": 10,
        "Aircore": 9,
    }
    STANDARD_PORT_NUM = 11       # port number the standard is run on.
    CAL_RUN_TYPES = {2}         # run_type_num values written to hats.calibrations
    WARMUP_RUN_TYPE = 3         # run type num warmup runs are on.
    EXCLUDE = []               # exclude from autoscaling
    BASE_MARKER_SIZE = 15

    MARKER_MAP = {
        1: 'o',    # Standard
        11: 'D',   # aircore
        12: '*',   # Cal
        13: 'o',
        14: 'o',
        15: 'o',
        16: 'o',
        17: 'o',
        18: 'o',
        19: 'o',
    }
    
    COLOR_MAP = {1: 'cornflowerblue', 11: 'orange', 12: 'darkgreen', 13: 'cyan', 14: 'pink',
                 15: 'gray', 16: 'blue', 17: 'red', 18: 'darkred', 19: 'lightgreen',
                 3: 'orange', 5: 'gray', 7: 'red'}  # ports used in 2021
    
    def __init__(self):
        super().__init__()
        self.inst_id = 'bld1'
        self.inst_num = self.INSTRUMENTS.get(self.inst_id)  # Lookup inst_num from INSTRUMENTS
        self.start_date = '20210906'         # data before this date is not used.
        self.gc_dir = Path("/hats/gc/bld1")
        self.export_dir = self.gc_dir / "results"
        self.response_type = 'height'

        self.analytes = {'SF6':6, 'N2O':5, 'CFC11':114, 'CFC12':22, 'CFC113':32, 'h1211':26}
        self.gas_chans = {
            'a': ['CFC11', 'CFC12', 'CFC113', 'h1211'],
            'b': ['SF6', 'N2O']
        }
        self.chan_map = {gas: ch for ch, gases in self.gas_chans.items() for gas in gases}

        self.molecules  = self.analytes.keys()
        self.analytes_inv = {v: k for k, v in self.analytes.items()}

        self.norm = Normalizing(self.inst_id, self.STANDARD_PORT_NUM, 'port', self.response_type)

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
                #AND detrend_method_num != 3
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

        # base port_idx logic for both
        df['port_idx'] = df['port'].astype(int)

        # Start with site-based colors
        df['port_color'] = df['port'].map(self.COLOR_MAP).fillna('gray')

        # port markers
        df['port_marker'] = df['port_idx'].map(self.MARKER_MAP)

        return df

    def load_calcurves(self, pnum, channel, earliest_run):
        """
        Returns the calibration curves from ng_response for a given parameter number and channel.
        
        NOTE: I updated load_calcurves for FE3 and may need to update it for BLD1 as well!!!
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
                and run_date >= '{earliest_run}'
            order by run_date desc;
        """
        df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            print(f"No calibration curves found for parameter {pnum} and channel '{channel}'.")
            return df
        df['flag'] = pd.to_numeric(df['flag'], errors='coerce').fillna(1).astype(int)
        return df
    
    def export_run(self, df, file):
        pass
                
    def export_run_alldata(self, df, file):
        
        start_date = df['run_time'].min().strftime('%Y-%m-%d %H:%M:%S')

        query = f"""
            SELECT analysis_datetime, run_time, port, port_info, parameter_num,
               height, mole_fraction, retention_time, flag FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND run_time = '{start_date}'
            ORDER BY analysis_datetime;
        """

        df = pd.DataFrame(self.db.doquery(query))

        # --- Map flag strings to boolean ---
        if 'flag' in df.columns:
            df['flag'] = df['flag'].apply(lambda x: False if str(x).strip() == '...' else True)

        self.export_run_legacy(df, file)

    def export_run_legacy(self, df, filepath):
        """
        Export the current run to a legacy CSV format, flattening analytes
        (SF6, N2O, CFC11, CFC12, CFC113, h1211) into columns based on parameter_num.
        Ensures one row per analysis_datetime.
        """

        #analytes = ['SF6', 'N2O', 'CFC11', 'CFC12', 'CFC113', 'h1211']
        analytes = self.analytes.keys()
        base_cols = ['time', 'port']  # <-- 'time' instead of 'analysis_time'
        export_cols = base_cols + [
            f"{mol}_{suffix}"
            for mol in analytes
            for suffix in ['ht', 'rt', 'value', 'flag']
        ]

        # --- Base table: one row per unique timestamp ---
        df_base = (
            df[['analysis_datetime', 'port']]
            .drop_duplicates(subset=['analysis_datetime'])
            .copy()
        )
        df_base['time'] = pd.to_datetime(df_base['analysis_datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_out = df_base[['time', 'port']].copy()

        # --- Add analyte data (pivot per analyte then merge) ---
        for mol, pnum in self.analytes.items():
            subset = df[df['parameter_num'] == pnum].copy()

            # ensure expected columns
            for col in ['height', 'retention_time', 'mole_fraction', 'flag']:
                if col not in subset.columns:
                    subset[col] = pd.NA if col != 'flag' else False

            # rename for clarity
            subset.rename(columns={
                'height': f'{mol}_ht',
                'retention_time': f'{mol}_rt',
                'mole_fraction': f'{mol}_value',
                'flag': f'{mol}_flag'
            }, inplace=True)

            # ensure datetime column for merge
            subset['analysis_dt'] = pd.to_datetime(subset['analysis_datetime'])
            df_out['analysis_dt'] = pd.to_datetime(df_out['time'])

            # reduce to one row per analysis_datetime
            subset = subset.groupby('analysis_dt', as_index=False).first()

            # merge safely
            df_out = df_out.merge(
                subset[['analysis_dt', f'{mol}_ht', f'{mol}_rt', f'{mol}_value', f'{mol}_flag']],
                on='analysis_dt',
                how='left'
            )

        # drop helper column
        df_out.drop(columns=['analysis_dt'], inplace=True, errors='ignore')

        # --- Ensure all expected columns exist ---
        for col in export_cols:
            if col not in df_out.columns:
                df_out[col] = False if col.endswith('_flag') else pd.NA

        # --- Reorder columns and save ---
        df_out = df_out[export_cols]
        df_out.to_csv(filepath, index=False)
        print(f"Legacy CSV saved: {filepath}")
