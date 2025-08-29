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
        """Returns a dictionary of analytes and parameter numbers. """
        sql = f"SELECT param_num, channel, display_name FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
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
    
    def qurey_return_scale_num(self, parameter_num):
        sql = f""" SELECT idx FROM reftank.scales where parameter_num = {parameter_num}; """
        r = self.db.doquery(sql)
        if not r:
            raise ValueError(f"No scale number found for parameter number {parameter_num}.")
        return r[0]['idx']  

    def query_return_run_list(self, runtype=None, start_date=None, end_date=None):
        """ Returns a list of run_times for a run_type_num and start/end range
            start_date and end_date should be in YYYY-MM-DD format """
        
        if runtype is None:
            rt = 'AND run_type_num <> 3'        # exclude warmup runs
        else:
            rt = f'AND run_type_num = {runtype}'

        if end_date is None:
            t1 = 'AND run_time <= UTC_TIMESTAMP()'
            now = datetime.now()
            end_date = now.strftime("%Y-%m-%d")
        else:
            t1 = f'AND run_time <= "{end_date}"'
            
        # 2 month window
        if start_date is None:
            t0 = f'AND run_time >= ("{end_date}" - INTERVAL 2 MONTH)'
        else:
            t0 = f'AND run_time >= "{start_date}"'
            
        sql = f"""
            SELECT DISTINCT run_time
            FROM hats.ng_data_processing_view
            WHERE inst_num = {self.inst_num}
                {rt}
                {t0}
                {t1}
            ORDER BY run_time;
        """
        
        df = pd.DataFrame(self.doquery(sql))
        if df.empty:
            return []
        df['run_time'] = pd.to_datetime(df['run_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df['run_time'].to_list()
        
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
        
    def scale_assignments(self, tank, pnum):
        """
        Returns a dictionary of scale values for a given tank and parameter number (pnum).
        """
        # Extract only the digits before the first "_" in the tank variable
        #match = re.search(r'(\d+)[^\d_]*_', tank)
        #tank = match.group(1) if match else ''.join(filter(str.isdigit, tank))
        
        sql = f"""
            SELECT start_date, serial_number, level, coef0, coef1, coef2 FROM hats.scale_assignments 
            #where serial_number like '%{tank}%'
            where serial_number = '{tank}'
            and inst_num = {self.inst_num} 
            and scale_num = (select idx from reftank.scales where parameter_num={pnum} and current=1);
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

    def scale_values(self, tank, pnum):
        """
        Returns a dictionary of scale values for a given tank and parameter number.
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

    def param_calcurves(self, df):
        """
        Returns the calibration curves for a given port number and channel.
        """
        print(df.columns)
        
        try:
            scale_num = int(df['cal_scale_num'].dropna().unique()[0])      # unique to each parameter_num
            channel = df['channel'].unique()[0]
        except (IndexError, ValueError):
            return pd.DataFrame()  # No valid scale_num or channel found
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

    def calc_mole_fraction(self, df):
        df = df.copy()
        cols = ["normalized_resp", "coef0", "coef1", "coef2", "coef3"]
        arr = df[cols].to_numpy()
        df["mole_fraction"] = [
            self.invert_poly_to_mf(y, a0, a1, a2, a3, mf_min=0.0, mf_max=3000)
            for (y, a0, a1, a2, a3) in arr
        ]
        return df['mole_fraction']

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

    # developed for fe3 -- needs work for other instruments
    def fit_poly(self, 
        df: pd.DataFrame,
        order: int = 2,
        xcol: str = "cal_mf",
        ycol: str = "normalized_resp",
        ports_col: str = "port",
        exclude_ports = (1, 9),
        flag_col: str = "flag",
        bad_flag: int = 1,
        extra_mask: pd.Series | None = None,
    ):
        """
        Fit ycol = poly(order)(xcol) using rows where:
        • port not in exclude_ports
        • flag_col != bad_flag
        • xcol and ycol are not null
        Returns (model, fitted_df, poly) where:
        • model: dict with coefs, r2, adj_r2, rmse, n, order
        • fitted_df: subset with columns [xcol, ycol, yhat, resid]
        • poly: np.poly1d, callable: y_pred = poly(x_new)
        """

        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2, or 3")

        # Build mask
        m = pd.Series(True, index=df.index)
        if ports_col in df:
            m &= ~df[ports_col].isin(exclude_ports)
        if flag_col in df:
            m &= (df[flag_col] != bad_flag)
        m &= df[xcol].notna() & df[ycol].notna()
        if extra_mask is not None:
            m &= extra_mask.reindex(df.index, fill_value=False)

        sub = (
            df.loc[m, [xcol, ycol]]
            .astype(float)
            .sort_values(xcol)
        )

        n = len(sub)
        if n < (order + 1):
            #raise ValueError(f"Need at least {order+1} points to fit a degree-{order} polynomial; got {n}.")
            return None, None, None

        x = sub[xcol].to_numpy()
        y = sub[ycol].to_numpy()

        # Fit
        coefs = np.polyfit(x, y, deg=order)  # [a_k ... a1, a0]
        poly = np.poly1d(coefs)
        yhat = poly(x)

        # Metrics
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
        p = order + 1  # number of parameters
        adj_r2 = np.nan if n <= p else 1 - (1 - r2) * (n - 1) / (n - p)
        rmse = np.sqrt(ss_res / max(n - p, 1))

        # Package outputs
        coef_dict = {f"coef{i}": float(c) for i, c in zip(range(order, -1, -1), coefs)}
        model = {
            "order": order,
            "coefs": coef_dict,       # coef{order}..coef0 (highest power → intercept)
            "coefs_array": coefs.tolist(),
            "r2": float(r2),
            "adj_r2": float(adj_r2),
            "rmse": float(rmse),
            "n": int(n),
        }
        fitted = sub.assign(yhat=yhat, resid=y - yhat)

        return model, fitted, poly
    
    def _fit_row_for_current_run(self, run, order=2):
        # Fit polynomial for the current run and return a dict suitable for inserting into ng_response.
        try:
            ref_tank = run.loc[run['port'] == self.STANDARD_PORT_NUM, 'port_info'].iat[0]
        except IndexError:
            return None
        model, _, _ = self.fit_poly(run, order=int(order))
        if model is None:
            return None
        return {
            "run_date": pd.to_datetime(run["run_time"].iat[0], utc=True).strftime('%Y-%m-%d %H:%M:%S'),
            "inst_num": self.inst_num,
            "site": "BLD",  # hardcoded for Boulder
            "scale_num": int(run["cal_scale_num"].iat[0]),
            "channel": run["channel"].iat[0],
            "serial_number": ref_tank,
            "coef3": float(model["coefs"].get("coef3", 0.0)),
            "coef2": float(model["coefs"].get("coef2", 0.0)),
            "coef1": float(model["coefs"].get("coef1", 0.0)),
            "coef0": float(model["coefs"].get("coef0", 0.0)),
            "function": "poly",
            "flag": 0,
        }


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
            std.groupby('run_time', group_keys=False)[['ts', self.response_type]]
            .apply(lambda seg: self._smooth_segment(seg, frac))
            .squeeze()
        )
        
        return std[['analysis_datetime','run_time','smoothed']]

    def merge_smoothed_data(self, df):
        # smoothed std or reference tank injection
        std = self.calculate_smoothed_std(df, min_pts=5, frac=0.3)

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
    
    RUN_TYPE_MAP = {
        "All": None,        # no filter
        "Flasks": 1,        # run_type_num
        #"Calibrations": 2,
        "PFPs": 5,
    }
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
                
    def load_data(self, pnum, channel=None, run_type_num=None, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            run_type_num (int, optional): Run type number to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        norm = Normalizing(self.inst_id, self.STANDARD_RUN_TYPE, 'run_type_num', self.response_type)
        
        if end_date is None:
            end_date = datetime.today()
        elif len(end_date) == 4:
            # check for YYMM format
            end_date = datetime.strptime(end_date, "%y%m")
            end_date = start_date.strftime("%Y-%m-31")    # end of the month
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

        print(f"Loading data from {start_date} to {str(end_date)} for parameter {pnum}")
        # todo: use flags - using low_flow flag
        query = f"""
            SELECT * FROM hats.ng_data_processing_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                #AND detrend_method_num != 3
                AND run_time BETWEEN '{start_date}' AND '{end_date}'
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
        
        # build a port_idx for plotting colors
        mask = df['run_type_num'].eq(5)     # pfp runtype
        base = pd.to_numeric(df['port'], errors='coerce').astype('float64')
        pfp  = pd.to_numeric(df['flask_port'], errors='coerce').astype('float64') + 20
        res = base.copy()
        res.loc[mask] = pfp.loc[mask]          # explicit assignment avoids where/mask downcast warning
        df['port_idx'] = res.round().astype('Int64')   # final, intentional cast to nullable int
        
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
    
    RUN_TYPE_MAP = {
        "All": None,        # no filter
        "Flasks": 1,        # run_type_num
        "Calibrations": 2,
        "Other": 4,
        #"PFPs": 5,
    }
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
    
    def load_data(self, pnum, channel=None, run_type_num=None, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            run_type_num (int, optional): Run type number to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        norm = Normalizing(self.inst_id, self.STANDARD_PORT_NUM, 'port', self.response_type)
        
        if end_date is None:
            end_date = datetime.today()
        elif len(end_date) == 4:
            # check for YYMM format
            end_date = datetime.strptime(end_date, "%y%m")
            end_date = start_date.strftime("%Y-%m-31")    # end of the month
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

        print(f"Loading data from {start_date} to {end_date} for parameter {pnum}")
        # todo: use flags
        query = f"""
            SELECT * FROM hats.ng_data_processing_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                {channel_str}
                {run_type_filter}
                AND height <> -999
                #AND detrend_method_num != 3
                AND run_time BETWEEN '{start_date}' AND '{end_date}'
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
        df = norm.merge_smoothed_data(df)
        #df['mole_fraction']     = self.calc_mole_fraction(df)
        df['port_idx']          = df['port'].astype(int) + df['flask_port'].fillna(0).astype(int)
        
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
            
    def load_calcurves(self, pnum, channel, earliest_run):
        """
        Returns the calibration curves from ng_response for a given parameter number and channel.
        """
        scale_num = self.qurey_return_scale_num(pnum)
        
        ch = ''
        if channel is not None:
            ch = f"and channel = '{channel}'"
                  
        sql = f"""
            SELECT run_date, serial_number, coef0, coef1, coef2, coef3, function, flag FROM hats.ng_response
                where inst_num = {self.inst_num}
                and scale_num = {scale_num}
                {ch}
                and run_date >= '{earliest_run}'
            order by run_date desc;
        """
        return pd.DataFrame(self.db.doquery(sql))
class BLD1_Instrument(HATS_DB_Functions):
    """ Class for accessing BLD1 (Stratcore) specific functions in the HATS database. """
    
    def __init__(self):
        super().__init__()
        self.inst_id = 'bld1'
        self.inst_num = 220
        self.start_date = '20191217'         # data before this date is not used.
        self.gc_dir = Path("/hats/gc/bld1")
        self.export_dir = self.gc_dir / "results"
