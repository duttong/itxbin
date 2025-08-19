import sys
import pandas as pd
import numpy as np
import re
from functools import cached_property
from pathlib import Path
from datetime import datetime, timedelta
import calendar
from statsmodels.nonparametric.smoothers_lowess import lowess

class LOGOS_Instruments:
    INSTRUMENTS = {'m4': 192, 'fe3': 193, 'bld1': 220} 

    def __init__(self):
        # gcwerks-3 path
        self.gcexport_path = "/hats/gc/gcwerks-3/bin/gcexport"
        

class HATS_DB_Functions(LOGOS_Instruments):
    """ Class for accessing HATS database functions related to instruments. 
        Tailored to works on 'next generation' or 'ng_' tables."""
        
    def __init__(self, inst_id=None):
        super().__init__()
        self.inst_id = inst_id or 'fe3'  # Default to 'fe3' if no inst_id is provided
        self.inst_num = self.INSTRUMENTS.get(self.inst_id)  # Lookup inst_num from INSTRUMENTS
        if self.inst_num is None:
            raise ValueError(f"Invalid instrument ID: {self.inst_id}")

        # database connection
        sys.path.append('/ccg/src/db/')
        import db_utils.db_conn as db_conn # type: ignore
        self.db = db_conn.HATS_ng()
        self.doquery = self.db.doquery

    def gml_sites(self):
        """ Returns a dictionary of site codes and site numbers from gmd.site."""
        sql = "SELECT num, code FROM gmd.site;"
        df = pd.DataFrame(self.doquery(sql))
        site_dict = dict(zip(df['code'], df['num']))
        return site_dict

    def query_analytes(self):
        """Returns a dictionary of analytes and parameter numbers."""
        sql = f"SELECT param_num, display_name FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
        df = pd.DataFrame(self.doquery(sql))
        analytes_dict = dict(zip(df['display_name'], df['param_num']))
        return analytes_dict
    
    def query_molecules(self):
        """ Returns a list of analytes or molecules (no parameter number) """
        analytes = self.query_analytes()
        return analytes.keys()
    
    def run_type_num(self):
        """ Run types defined in the hats.ng_run_types table """
        sql = "SELECT * FROM hats.ng_run_types;"
        r = self.doquery(sql)
        results = {entry['name'].lower(): entry['num'] for entry in r}
        results['cal'] = 2
        results['std'] = 8
        return results

    def standards(self):
            """ Returns a dictionary of standards files and a key used in the HATS db."""
            sql = "SELECT num, serial_number, std_ID FROM hats.standards"
            df = pd.DataFrame(self.doquery(sql))
            standards_dict = df.set_index('std_ID')[['num', 'serial_number']].T.to_dict('list')
            return standards_dict
        
    def scale_number(self, parameter_num):
        sql = f"""
            select idx from reftank.scales where parameter_num = {parameter_num}
        """
        r = self.db.doquery(sql)
        if not r:
            raise ValueError(f"No scale number found for parameter number {parameter_num}.")
        return r[0]['idx']
        
    def scale_values(self, tank, pnum):
        """
        Returns a dictionary of scale values for a given tank and parameter number (pnum).
        """
        # Extract only the digits before the first "_" in the tank variable
        match = re.search(r'(\d+)[^\d_]*_', tank)
        tank = match.group(1) if match else ''.join(filter(str.isdigit, tank))
        
        sql = f"""
            SELECT start_date, serial_number, level, coef0, coef1, coef2 FROM hats.scale_assignments 
            where serial_number like '%{tank}%'
            and inst_num = {self.inst_num} 
            and scale_num = (select idx from reftank.scales where parameter_num = {pnum});
        """
        df = pd.DataFrame(self.db.doquery(sql))
        if not df.empty:
            return df.iloc[0].to_dict()
        else:
            Warning(f"Scale values not found for tank {tank} and parameter number {pnum}.")
            return None

    def return_analysis_nums(self, df, time_col='dt_run'):
        """ Inserts the analysis numbers into the dataframe based on the time column.
        This function assumes that the time_col in df is in a datetime format.
        Args:
            df (pd.DataFrame): DataFrame containing the time column.
            time_col (str): Name of the column in df that contains the run times.
        Returns:
            pd.DataFrame: DataFrame with an additional column 'analysis_num' containing the analysis numbers.
        """
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in the DataFrame.")
        if df.empty:
            df['analysis_num'] = None
            return df
        
        # Copy and ensure df[time_col] is datetime64
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        # If df is empty, just add the column and return
        if df.empty:
            df['analysis_num'] = None
            return df

        # Determine the min/max run times so we only pull what's needed
        min_time = df[time_col].min().strftime('%Y-%m-%d %H:%M:%S')
        max_time = df[time_col].max().strftime('%Y-%m-%d %H:%M:%S')
    
        sql = (
            "SELECT analysis_time, num "
            "FROM hats.ng_analysis "
            f"WHERE inst_num = {self.inst_num} "
            f"  AND analysis_time >= '{min_time}' "
            f"  AND analysis_time <= '{max_time}';"
        )

        db_df = pd.DataFrame(self.db.doquery(sql))

        # Now merge back onto the original df:
        out = df.merge(
            db_df,
            how='left',
            left_on=time_col,
            right_on='analysis_time'
        ).rename(columns={'num': 'analysis_num'})

        out.drop(columns=['analysis_time'], inplace=True)
        return out

    def merge_calibration_tank_values(self, df):
        """ Load calibration data for the given parameter number (pnum) and merge it with the provided dataframe (df).
            Also merges the scale number.
            This function assumes df has a 'port_info' column containing serial numbers.
            ** Currently does not handle M4 data. The port_info column is much more complex in M4.
        """
        df = df.copy()
        pnum = df['parameter_num'].iat[0]
        
        # Build a clean list of cal serial numbers from the dataframe
        cal_serials = (
            df['port_info']
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s.ne('')]
            .unique()
            .tolist()
        )

        if not cal_serials:
            # Nothing to filter by; choose what you want to do here
            result = []  # or return empty DataFrame
        else:
            placeholders = ','.join(['%s'] * len(cal_serials))
            sql = f"""
                SELECT sa.serial_number, sa.scale_num, sa.start_date, sa.tzero, sa.coef0, sa.coef1, sa.coef2, sa.assign_date
                FROM hats.scale_assignments AS sa
                JOIN reftank.scales AS sc
                ON sa.scale_num = sc.idx
                WHERE sa.inst_num = %s
                AND sc.parameter_num = %s
                AND sa.serial_number IN ({placeholders});
            """
            params = [self.inst_num, pnum, *cal_serials]
            result = pd.DataFrame(self.db.doquery(sql, params))

        if result.empty:
            # returns NaN for cal_mf if no calibration data is found
            #print(f"No calibration data found for parameter {pnum} and serial numbers: {cal_serials}")
            df['cal_mf'] = np.nan
            df['scale_num'] = self.scale_number(pnum)
            return df
                
        cols = ['serial_number', 'scale_num', 'coef0']  # add coef1, coef2... if needed
        res = result[cols].copy()

        df = (df.assign(port_info=df['port_info'].astype(str).str.strip())
                .merge(res.rename(columns={'serial_number': 'port_info'}),
                    on='port_info', how='left'))
        df.rename(columns={'coef0': 'cal_mf'}, inplace=True)
        df['cal_mf'] = pd.to_numeric(df['cal_mf'], errors='coerce')
        
        # missing scale_num added it
        df.loc[df['scale_num'].isna(), 'scale_num'] = self.scale_number(pnum)
        return df

    def param_calcurves(self, df):
        """
        Returns the calibration curves for a given port number and channel.
        """
        scale_num = int(df['scale_num'].dropna().unique()[0])      # unique to each parameter_num
        channel = df['channel'].unique()[0]
        earliest_run = df['run_time'].min() - pd.DateOffset(years=1)  # 1 year before the earliest run
        
        if scale_num is None or channel is None:
            raise ValueError("Scale number and channel must be specified.")
        if not isinstance(scale_num, int):
            raise ValueError("Scale number must be an integer.")
        if not isinstance(channel, str):
            raise ValueError("Channel must be a string.")
        if not isinstance(earliest_run, pd.Timestamp):
            raise ValueError("Earliest run must be a pandas Timestamp.")
            
        sql = f"""
            SELECT run_date, serial_number, coef0, coef1, coef2, coef3, function, flag 
            FROM hats.ng_response
                where inst_num = {self.inst_num}
                and scale_num = {scale_num}
                and channel = '{channel}'
                and run_date >= '{earliest_run}'
            order by run_date desc;
        """
        df = pd.DataFrame(self.db.doquery(sql))
        # ng_response table has run_date and ng_data_view uses run_time
        df.rename(columns={'run_date': 'run_time'}, inplace=True)
        return df

    def select_cal_and_compute_mf(
        self,
        df: pd.DataFrame,
        curves: pd.DataFrame,
        resp_col: str = "normalized_resp",
        by: list | None = None,           # e.g., ['channel','serial_number'] if you keep multiple families
        flag_col: str = "flag",
        flagged_val: int = 1,
    ) -> pd.DataFrame:
        """
        For each row in df, attach the latest unflagged calibration with cal.run_time <= df.run_time,
        then invert y = c0 + c1*x + c2*x^2 + c3*x^3 to get x = mf_calc from y=df[resp_col].

        Returns a new DataFrame with columns: coef0..coef3, function (if present), and mf_calc.
        Rows with no eligible older calibration (or non-invertible polynomials) get mf_calc = NaN.
        """
        if resp_col not in df.columns:
            raise KeyError(f"df is missing '{resp_col}'")

        out = df.copy()
        curves = curves.copy()
        
        out['run_time']    = self._ensure_utc(out['run_time'])
        curves['run_time'] = self._ensure_utc(curves['run_time'])

        # Keep unflagged calibrations (flag != flagged_val OR NaN)
        cal = curves.loc[curves[flag_col].ne(flagged_val) | curves[flag_col].isna(),
                        ['run_time','serial_number','coef0','coef1','coef2','coef3'] + (['function'] if 'function' in curves else [])]

        # Sort for merge_asof
        by = list(by) if by else []
        cal = cal.sort_values(by + ['run_time']).reset_index(drop=True)

        # Build one row per (by..., run_time) to resolve calibration once per run group (fast)
        unique_runs = out.drop_duplicates(subset=by + ['run_time'])[[*by, 'run_time']].sort_values(by + ['run_time'])

        # As-of merge: latest cal where cal.run_time <= run.run_time, respecting 'by' partitions if any
        cal_for_run = pd.merge_asof(
            unique_runs,
            cal,
            on='run_time',
            by=by if by else None,
            direction='backward',
            allow_exact_matches=True
        )

        # Attach coefficients back to every row by (by..., run_time)
        out = out.merge(cal_for_run, on=[*by, 'run_time'], how='left', validate='many_to_one')

        # --- Invert polynomial: resp -> mf ----------------------------------------------------------
        r  = out[resp_col].astype(float)
        c0 = out['coef0']
        c1 = out['coef1']
        c2 = out['coef2']
        c3 = out['coef3']

        out['mole_fraction'] = [
            self.invert_poly_to_mf(y, a0, a1, a2, a3)
            for y, a0, a1, a2, a3 in zip(r, c0, c1, c2, c3)
        ]

        return out

    # Helper: invert a single polynomial
    @staticmethod
    def invert_poly_to_mf(y, a0, a1, a2, a3, mf_min=0.0, mf_max=None):
        # No calibration? (all NaN) or all-zero coefs -> NaN
        if pd.isna(a0) and pd.isna(a1) and pd.isna(a2) and pd.isna(a3):
            return np.nan
        if (a0 == 0 or pd.isna(a0)) and (a1 == 0 or pd.isna(a1)) and (a2 == 0 or pd.isna(a2)) and (a3 == 0 or pd.isna(a3)):
            return np.nan

        # Treat near-zero with tolerance
        z1 = np.isclose(a1, 0.0, rtol=0, atol=1e-14)
        z2 = np.isclose(a2, 0.0, rtol=0, atol=1e-14)
        z3 = np.isclose(a3, 0.0, rtol=0, atol=1e-14)

        # Linear: y = a0 + a1*x
        if z2 and z3:
            if np.isclose(a1, 0.0, atol=1e-14):
                return np.nan
            x = (y - a0) / a1
            return x if (x >= mf_min) and (mf_max is None or x <= mf_max) else np.nan

        # Quadratic: y = a0 + a1*x + a2*x^2
        if z3:
            A, B, C = a2, a1, (a0 - y)
            if np.isclose(A, 0.0, atol=1e-14):  # fallback to linear if a2 ~ 0
                if np.isclose(B, 0.0, atol=1e-14):
                    return np.nan
                x = -C / B
                return x if (x >= mf_min) and (mf_max is None or x <= mf_max) else np.nan
            disc = B*B - 4*A*C
            if disc < 0:
                return np.nan
            sqrt_disc = np.sqrt(disc)
            r1 = (-B + sqrt_disc) / (2*A)
            r2 = (-B - sqrt_disc) / (2*A)
            candidates = [r for r in (r1, r2)
                        if (r >= mf_min) and (mf_max is None or r <= mf_max)]
            if not candidates:
                return np.nan
            # Heuristic: pick the smaller non-negative (often right for concave response curves)
            return min(candidates)

        # Cubic: y = a0 + a1*x + a2*x^2 + a3*x^3  ->  a3*x^3 + a2*x^2 + a1*x + (a0 - y) = 0
        coeffs = [a3, a2, a1, a0 - y]
        roots = np.roots(coeffs)
        real_roots = [float(np.real(z)) for z in roots if np.isreal(z)]
        # Filter by domain
        real_roots = [x for x in real_roots if (x >= mf_min) and (mf_max is None or x <= mf_max)]
        if not real_roots:
            return np.nan
        # Heuristic: smallest non-negative root is usually the physical solution domain for MF
        return min(real_roots)
                
    @staticmethod
    def _ensure_utc(s: pd.Series) -> pd.Series:
        # Coerce to datetime first
        s = pd.to_datetime(s, errors='coerce')
        # If tz-aware, convert to UTC; if naive, localize as UTC
        if pd.api.types.is_datetime64tz_dtype(s):
            return s.dt.tz_convert('UTC')
        else:
            return s.dt.tz_localize('UTC')

class Normalizing():
    
    def __init__(self, inst_id, std_run_type, run_type_column, response_type='area'):
        self.inst_id = inst_id
        self.run_type_column = run_type_column
        self.standard_run_type = std_run_type
        self.response_type = response_type

    def _smooth_segment(self, seg, frac):
        # skip tiny segments
        if len(seg) < 3 or seg['ts'].max() == seg['ts'].min():
            return pd.Series(seg[self.response_type].values, index=seg.index)
        # do LOWESS
        return pd.Series(
            lowess(seg[self.response_type], seg['ts'], frac=frac, return_sorted=False),
            index=seg.index
        )
    
    def calculate_smoothed_std(self, df, min_pts=8, frac=0.4):
        """ Calculate smoothed standard deviation for the standard run type.
            This function uses LOWESS smoothing on the area data.
            min_pts is the minimum number of points required to perform smoothing.
            frac is the fraction of points used for smoothing.
            The smoothed values are returned in a new column 'smoothed'.
        """
        std = (
            df.loc[df[self.run_type_column] == self.standard_run_type,
                    ['analysis_datetime', 'run_time', 'detrend_method_num', self.response_type]]
                .dropna()
                .sort_values('analysis_datetime')
                .copy()
        )
        
        # keep only those rows *after* the first in each run_time
        # this is to avoid smoothing the first point in each run_time which is often an outlier
        if self.inst_id == 'm4':
            std = std[std.groupby('run_time').cumcount() > 0].copy()
        
        # Not enough points to smooth
        if len(std) < min_pts:
            std['smoothed'] = np.nan
            return std[['analysis_datetime','run_time','smoothed']]

        std['ts'] = std['analysis_datetime'].astype(np.int64) // 10**9
        
        detrend_method = std['detrend_method_num'].iat[0]

        if detrend_method == 1:
            # point to point is the same as a small frac for LOWESS
            frac = 0.01

        std['smoothed'] = (
            std
            .groupby('run_time', group_keys=False)[['ts', self.response_type]]
            .apply(lambda seg: self._smooth_segment(seg, frac))
        )

        return std[['analysis_datetime','run_time','smoothed']]

    def merge_smoothed_data(self, df):
        # smoothed std or reference tank injection
        std = self.calculate_smoothed_std(df, min_pts=5, frac=0.4)

        out = (
            df
            .merge(std, on=['analysis_datetime','run_time'], how='left')
            .sort_values('analysis_datetime')
        )

        # explicitly select just the 'smoothed' column for the group operation which is the std
        out['smoothed'] = (
            out
            .groupby('run_time', group_keys=False)['smoothed']
            .apply(lambda s: s
                .interpolate(method='linear', limit_direction='both')
                .ffill()
                .bfill()
            )
        )

        out['normalized_resp'] = out[self.response_type] / out['smoothed']
                
        return out

class M4_Instrument(HATS_DB_Functions):
    """ Class for accessing M4 specific functions in the HATS database. """
    
    STANDARD_RUN_TYPE = 8
    COLOR_MAP_RUN_TYPE = {
        1: "#1f77b4",  # Flask
        4: "#ff7f0e",  # Other
        5: "#2ca02c",  # PFP
        6: "#dd89f9",  # Zero
        7: "#c7811b",  # Tank
        8: "#505c5c",  # Standard
        "Response": "#e04c19",  # Response
        "Ratio": "#1f77b4",  # Ratio
        "Mole Fraction": "#2ca02c",  # Mole Fraction
    }
    
    COLOR_MAP = {
        # SSV ports (0-16)
        0: 'cornflowerblue', 1: 'green', 2: 'red', 3: 'cyan', 4: 'hotpink',
        5: 'purple', 6: 'orange', 7: 'darkgreen', 8: 'darkred', 9: 'lightgreen',
        10: 'cornflowerblue', 11: 'green', 12: 'red', 13: 'cyan', 14: 'pink',
        15: 'teal', 16: 'orange',
        # PFPs (20-32)
        20: 'cornflowerblue', 21: 'green', 22: 'red', 23: 'cyan', 24: 'hotpink',
        25: 'purple', 26: 'orange', 27: 'darkgreen', 28: 'darkred', 29: 'lightgreen',
        30: 'black', 31: 'coral', 32: 'lightblue'}
    
    def __init__(self):
        super().__init__()
        self.inst_id = 'm4'
        self.inst_num = 192
        self.start_date = '20231223'         # data before this date is not used.
        self.gc_dir = Path("/hats/gc/m4")
        self.export_dir = self.gc_dir / "results"

        self.molecules = self.query_molecules()
        self.analytes = self.query_analytes()
        self.analytes_inv = {int(v): k for k, v in self.analytes.items()}
        self.response_type = 'area'
                
    def load_data(self, pnum, channel=None, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        norm = Normalizing(self.inst_id, self.STANDARD_RUN_TYPE, 'run_type_num', self.response_type)
        
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = datetime.strptime(end_date, "%y%m")
        last_day = calendar.monthrange(end_date.year, end_date.month)[1]
        end_date = end_date.replace(day=last_day)

        if start_date is None:
            start_date = end_date - timedelta(days=60)
        else:
            start_date = datetime.strptime(start_date, "%y%m")

        start_date_str = start_date.strftime("%Y-%m-01")
        end_date_str = end_date.strftime("%Y-%m-%d")

        print(f"Loading data from {start_date_str} to {str(end_date_str)} for parameter {pnum}")
        # todo: use flags - using low_flow flag
        query = f"""
            SELECT analysis_datetime, run_time, sample_datetime, run_type_num, port, port_info, flask_port, detrend_method_num, 
                area, mole_fraction, net_pressure, flag, sample_id, pair_id_num, site
            FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                #AND area != 0
                AND detrend_method_num != 3
                #AND low_flow != 1
                AND run_time BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            print(f"No data found for parameter {pnum} in the specified date range.")
            self.data = pd.DataFrame()
            return
        
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
        df = norm.merge_smoothed_data(df)
        df['parameter_num']     = pnum
        df['port_idx']          = df['port'].astype(int)        # used for plotting
        #df = self.merge_calibration_tank_values(df)   # add calibration tank mole fractions

        df.loc[df['run_type_num'] == 5, 'port_idx'] = (
            df.loc[df['run_type_num'] == 5, 'flask_port'] + 20      # PFP ports are offset by 20
        )
        
        df = self.add_port_labels(df)
        
        self.data = df.sort_values('analysis_datetime')
        return self.data
        
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
        df.loc[mask, 'port_label'] = (
            df.loc[mask, 'site'] + ' ' +
            df.loc[mask, 'sample_id'].astype(str) + ' (' +
            df.loc[mask, 'flask_port'].astype(int).astype(str) + ')'
        )

        # clean up any stray spaces
        df['port_label'] = df['port_label'] \
                            .str.replace(r'\s+', ' ', regex=True) \
                            .str.strip()
        
        return df            

    def load_calcurves(self, df):
        """ Calcurves are not stored in ng_response for M4. """
        pass

class FE3_Instrument(HATS_DB_Functions):
    """ Class for accessing M4 specific functions in the HATS database. """
    
    STANDARD_PORT_NUM = 1       # port number the standard is run on.
    WARMUP_RUN_TYPE = 3         # run type num warmup runs are on.
    # color map made for a combination of SSV and Flask ports.
    COLOR_MAP = {
        # SSV ports (0-9)
        0: 'cornflowerblue', 1: 'green', 2: 'red', 3: 'cyan', 4: 'pink',
        5: 'gray', 6: 'orange', 7: 'darkgreen', 8: 'darkred', 9: 'lightgreen',
        # Flask ports (10-19)
        10: 'cornflowerblue', 11: 'blue', 12: 'red', 13: 'cyan', 14: 'pink',
        15: 'gray', 16: 'orange', 17: 'darkgreen', 18: 'darkred', 19: 'purple'}
    
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

    def return_preferred_channel(self, gas: str) -> str | None:
        gas_l = gas.lower()
        # override for CFC11/113
        if gas_l in ('cfc11','cfc113'):
            return 'c'
        # otherwise lookup in the precomputed map
        return self.molecule_channel_map.get(gas_l)
    
    def load_data(self, pnum, channel=None, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        norm = Normalizing(self.inst_id, self.STANDARD_PORT_NUM, 'port', self.response_type)
        
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = datetime.strptime(end_date, "%y%m")
        last_day = calendar.monthrange(end_date.year, end_date.month)[1]
        end_date = end_date.replace(day=last_day)

        if start_date is None:
            start_date = end_date - timedelta(days=60)
        else:
            start_date = datetime.strptime(start_date, "%y%m")

        start_date_str = start_date.strftime("%Y-%m-01")
        end_date_str = end_date.strftime("%Y-%m-%d")

        print(f"Loading data from {start_date_str} to {end_date_str} for parameter {pnum}")
        # todo: use flags
        channel_str = f"AND channel = '{channel}'" if channel else ""
        query = f"""
            SELECT analysis_datetime, run_time, run_type_num, port, port_info, flask_port, detrend_method_num, 
                height, mole_fraction, channel, flag, sample_id, pair_id_num, site
            FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                {channel_str}
                AND height != 0
                AND run_type_num != {self.WARMUP_RUN_TYPE}
                #AND detrend_method_num != 3
                AND run_time BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            print(f"No data found for parameter {pnum} in the specified date range.")
            self.data = pd.DataFrame()
            return
        
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'], errors='raise', utc=True)
        df['run_time']          = pd.to_datetime(df['run_time'], errors='raise', utc=True)
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].astype(int)
        df['height']            = df['height'].astype(float)
        df['mole_fraction']     = df['mole_fraction'].astype(float)
        df = norm.merge_smoothed_data(df)
        df['parameter_num']     = pnum
        df['port_idx']          = df['port'].astype(int) + df['flask_port'].fillna(0).astype(int)
        df = self.merge_calibration_tank_values(df)   # add calibration tank mole fractions
        
        df = self.add_port_labels(df)

        self.data = df.sort_values('analysis_datetime')
        return self.data
        
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

        return df
    
    def load_calcurves(self, df):
        """
        Returns the calibration curves from ng_response for a given port number and channel.
        """
        NUMTYPE_CALIBRATION = 2  # run_type_num for calibration runs
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot load calibration curves.")
        if 'run_type_num' not in df.columns:
            raise ValueError("DataFrame must contain 'run_type_num' column.")
        
        cal_runs = self.data.loc[self.data['run_type_num'] == NUMTYPE_CALIBRATION, 'run_time'].unique()
        cdf = self.data.loc[self.data['run_time'] == cal_runs[0]]

        scale_num = int(cdf['scale_num'].dropna().unique()[0])      # unique to each parameter_num
        channel = cdf['channel'].unique()[0]
        earliest_run = cdf['run_time'].min() - pd.DateOffset(years=1)  # 1 year before the earliest run
        
        if scale_num is None or channel is None:
            raise ValueError("Scale number and channel must be specified.")
        if not isinstance(scale_num, int):
            raise ValueError("Scale number must be an integer.")
        if not isinstance(channel, str):
            raise ValueError("Channel must be a string.")
        if not isinstance(earliest_run, pd.Timestamp):
            raise ValueError("Earliest run must be a pandas Timestamp.")
            
        sql = f"""
            SELECT run_date, serial_number, coef0, coef1, coef2, coef3, function, flag FROM hats.ng_response
                where inst_num = {self.inst_num}
                and scale_num = {scale_num}
                and channel = '{channel}'
                and run_date >= '{earliest_run}'
            order by run_date desc;
        """
        return pd.DataFrame(self.db.doquery(sql))
    
class BLD1_Instrument(HATS_DB_Functions):
    """ Class for accessing BLD1 (Stratcore) specific functions in the HATS database. """
    
    def __init__(self):
        super().__init__()
        self.inst_id = 'bld1'
        self.inst_num = 999
        self.start_date = '20191217'         # data before this date is not used.
        self.gc_dir = Path("/hats/gc/bld1")
        self.export_dir = self.gc_dir / "results"
