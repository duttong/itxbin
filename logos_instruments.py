import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from functools import cached_property
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
from statsmodels.nonparametric.smoothers_lowess import lowess

class LOGOS_Instruments:
    INSTRUMENTS = {'m4': 192, 'fe3': 193, 'bld1': 220} 
    
    LOGOS_sites = ['SUM', 'PSA', 'SPO', 'SMO', 'AMY', 'MKO', 'ALT', 'CGO', 'NWR',
            'LEF', 'BRW', 'RPB', 'KUM', 'MLO', 'WIS', 'THD', 'MHD', 'HFM',
            'BLD', 'MKO']
    
    BASE_MARKER_SIZE = 60   # for scatter plots. Can override in subclasses.

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
        df[time_col] = pd.to_datetime(df[time_col], utc=True)

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
        db_df['analysis_time'] = pd.to_datetime(db_df['analysis_time'], utc=True)

        # Now merge back onto the original df:
        out = df.merge(
            db_df,
            how='left',
            left_on=time_col,
            right_on='analysis_time'
        ).rename(columns={'num': 'analysis_num'})

        out.drop(columns=['analysis_time'], inplace=True)
        return out

    def upsert_mole_fractions(self, df, response_id=None):
        """
        Inserts or updates rows in hats.ng_mole_fractions using a batch upsert.
        Any non‑numeric mole_fraction (including blank strings) becomes NULL.
        """
        sql_insert = """
            INSERT INTO hats.ng_mole_fractions (
                analysis_num,
                parameter_num,
                ng_response_id,
                detrend_method_num,
                channel,
                mole_fraction,
                flag
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                ng_response_id = VALUES(ng_response_id),
                mole_fraction = VALUES(mole_fraction),
                flag = VALUES(flag),
                detrend_method_num = VALUES(detrend_method_num);         
        """

        # Coerce everything to float, invalid parses → NaN, then round
        df = df.copy()
        df['mole_fraction'] = (
            pd.to_numeric(df['mole_fraction'], errors='coerce')
            .replace([np.inf, -np.inf], np.nan)
            .round(5)
        )

        params = []
        for _, row in df.iterrows():
            # Convert pandas NaN → Python None so INSERT writes a NULL
            mf = row.mole_fraction
            mole_fraction = None if pd.isna(mf) else float(mf)
            id = 0 if response_id is None else response_id

            params.append((
                row.analysis_num,
                row.parameter_num,
                id,
                row.detrend_method_num,
                row.channel,
                mole_fraction,
                row.data_flag
            ))

            # flush batch if doMultiInsert returns True
            if self.db.doMultiInsert(sql_insert, params):
                params = []

        # any trailing rows
        if params:
            self.db.doMultiInsert(sql_insert, params, all=True)
        
    def update_flags_all_analytes(self, df):
        """Duplicate flags from df to all rows with the same analysis_num."""

        run_time = df.run_time.iat[0]

        # 1. Clear all flags for this run_time
        sql_clear = f"""
            UPDATE hats.ng_mole_fractions mf
            JOIN hats.ng_analysis a ON mf.analysis_num = a.num
            SET mf.flag = '...'
            WHERE a.inst_num = {self.inst_num}
            AND a.run_time = %s;
        """
        self.db.doquery(sql_clear, [run_time])

        # 2. Apply flagged values ('M..') to the appropriate analysis_nums
        flagged = df.loc[df['data_flag'] == 'M..']
        if not flagged.empty:
            analysis_nums = flagged['analysis_num'].unique().tolist()
            sql_set = f"""
                UPDATE hats.ng_mole_fractions mf
                JOIN hats.ng_analysis a ON mf.analysis_num = a.num
                SET mf.flag = 'M..'
                WHERE a.inst_num = {self.inst_num}
                AND a.run_time = %s
                AND mf.analysis_num IN ({','.join(['%s'] * len(analysis_nums))});
            """
            params = [run_time] + analysis_nums
            self.db.doquery(sql_set, params)
            
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

    def qurey_return_scale_num(self, pnum):
        """ Query the scale number for a given parameter number. """
        sql = f"""
            SELECT idx FROM reftank.scales 
            WHERE parameter_num = {pnum}
            AND current = 1;
        """
        df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            raise ValueError(f"Scale number not found for parameter number {pnum}.")
        return int(df['idx'].iat[0])

    def calc_mole_fraction(self, df):
        """ Wrapper to call the appropriate mole fraction calculation method
            based on the instrument type.
            Returns a DataFrame with an additional 'mole_fraction' column.
        """
        if self.inst_id == 'm4':
            return self.calc_mole_fraction_scalevalues(df)
        elif self.inst_id == 'fe3' or self.inst_id == 'bld1':
            return self.calc_mole_fraction_response(df)
        else:
            raise NotImplementedError(f"Mole fraction calculation not implemented for instrument '{self.inst_id}'.")
        
    def calc_mole_fraction_scalevalues(self, df):
        """
        Compute mole_fraction = (a0 + a1·days_elapsed) * x
        where days_elapsed is days since 1900-01-01 relative to run_time,
        and x is normalized_resp.
        This method uses scale values from the database.
        
        M4 uses this method.
        """
        pnum     = df['parameter_num'].iat[0]
        baseline = pd.Timestamp('1900-01-01', tz='UTC')
        mf       = pd.Series(index=df.index, dtype=float)

        # cache for scale values keyed by ref_tank
        scale_cache = {}

        for rt, grp in df.groupby('run_time'):
            mask = grp['run_type_num'] == self.STANDARD_RUN_TYPE
            if not mask.any():
                mf.loc[grp.index] = np.nan
                continue

            ref_tank = grp.loc[mask, 'port_info'].iat[0]

            # only call scale_values once per tank
            if ref_tank not in scale_cache:
                scale_cache[ref_tank] = self.scale_values(ref_tank, pnum)
            coefs = scale_cache[ref_tank]

            if coefs is None:
                mf.loc[grp.index] = np.nan
                continue

            a0 = float(coefs['coef0'])
            a1 = float(coefs['coef1'])
            days = (pd.to_datetime(rt) - baseline).days

            mf.loc[grp.index] = (a0 + a1 * days) * grp['normalized_resp']

        out = df.copy()
        out['mole_fraction'] = mf
        return out
    
    def calc_mole_fraction_response(self, df):
        """ This method uses the polynomial coefficients from the ng_response table.
            It can handle linear, quadratic, and cubic polynomials.
            It inverts the polynomial to solve for mole_fraction given normalized_resp.
            
            FE3 uses this method.
        """
        df = df.copy()
        cols = ["normalized_resp", "coef0", "coef1", "coef2", "coef3"]
        arr = df[cols].to_numpy()
        df["mole_fraction"] = [
            self.invert_poly_to_mf(y, a0, a1, a2, a3, mf_min=-20.0, mf_max=3000)
            for (y, a0, a1, a2, a3) in arr
        ]
        return df

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
        
        model, _, _ = self.fit_poly(run, order=int(order), flag_col='data_flag_int', bad_flag=1)
        if model is None:
            print('Not enough points to fit polynomial.')
            return None
        return {
            "run_date": pd.to_datetime(run["run_time"].iat[0], utc=True).strftime('%Y-%m-%d %H:%M:%S'),
            "inst_num": self.inst_num,
            "site": "BLD",  # hardcoded for Boulder
            "scale_num": int(0 if pd.isna(run["cal_scale_num"].iat[0]) else run["cal_scale_num"].iat[0]),
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

    def calculate_smoothed_std(
        self,
        df,
        min_pts: int = 8,
        verbose: bool = False,
        override_detrend_method_num: int | None = None,
    ):
        """
        Calculate smoothed standard deviation per run_time.
        Each run_time can have its own detrend_method_num controlling the smoothing,
        or you can override all of them with override_detrend_method_num.

        detrend_method_num options (least → most smoothing):
          1: point-to-point linear interpolation
          3: 2-point moving average
          2: LOWESS using roughly 5 points
          4: 3-point boxcar mean
          6: 5-point boxcar mean
          5: LOWESS using roughly 10 points
        """

        std = (
            df.loc[
                (df[self.run_type_column] == self.standard_run_type)
                & (df['data_flag'].eq('...')),
                ['analysis_datetime', 'run_time', 'detrend_method_num', self.response_type]
            ]
            .dropna()
            .sort_values('analysis_datetime')
            .copy()
        )
        std = std.reset_index(drop=True)

        # skip first points for M4
        #if self.inst_id == 'm4':
        #    std = std[std.groupby('run_time').cumcount() > 0].reset_index(drop=True)

        if len(std) < min_pts:
            std['smoothed'] = np.nan
            return std[['analysis_datetime', 'run_time', 'smoothed']]

        std['ts'] = std['analysis_datetime'].astype(np.int64) // 10**9
        smoothed = np.full(len(std), np.nan, dtype=float)

        lowess_points = {2: 5, 5: 10}
        boxcar_windows = {4: 3, 6: 5}
        moving_avg_windows = {3: 2}

        # group-level loop, but minimal overhead
        for run_time, seg in std.groupby('run_time', sort=False):

            if override_detrend_method_num is not None:
                detrend_method = override_detrend_method_num
            else:
                detrend_method = seg['detrend_method_num'].iloc[0]

            detrend_method = 2 if pd.isna(detrend_method) else int(detrend_method)
            if detrend_method not in (1, 2, 3, 4, 5, 6):
                detrend_method = 2

            seg = seg.sort_values('ts')
            t0 = time.time() if verbose else None

            if (
                detrend_method in lowess_points
                and len(seg) >= 3
                and seg['ts'].max() > seg['ts'].min()
            ):
                points = lowess_points[detrend_method]
                frac = min(points / len(seg), 1.0)
                y_smooth = lowess(
                    seg[self.response_type],
                    seg['ts'],
                    frac=frac,
                    return_sorted=False
                )
                method_desc = f"lowess_{points}pt_frac{frac:.3f}"
            elif detrend_method in boxcar_windows:
                window = boxcar_windows[detrend_method]
                y_smooth = (
                    seg[self.response_type]
                    .rolling(window=window, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )
                method_desc = f"boxcar_{window}pt"
            elif detrend_method in moving_avg_windows:
                window = moving_avg_windows[detrend_method]
                y_smooth = (
                    seg[self.response_type]
                    .rolling(window=window, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )
                method_desc = f"moving_avg_{window}pt"
            else:
                y_smooth = (
                    seg.set_index('ts')[self.response_type]
                    .interpolate(method='index', limit_direction='both')
                    .to_numpy()
                )
                method_desc = "linear_interp"

            smoothed[seg.index] = y_smooth

            if verbose:
                print(
                    f"run_time={run_time} | n={len(seg)} | detrend={detrend_method} "
                    f"| method={method_desc} | time={time.time()-t0:.4f}s"
                )

        std['smoothed'] = smoothed
        return std[['analysis_datetime', 'run_time', 'smoothed']]

    def merge_smoothed_data(
        self,
        df: pd.DataFrame,
        detrend_method_num: int | None = None,
    ) -> pd.DataFrame:
        """Merge smoothed standard data and calculate normalized response."""
        if 'smoothed' in df.columns:
            df = df.drop(columns=['smoothed'])

        std = self.calculate_smoothed_std(
            df,
            min_pts=5,
            override_detrend_method_num=detrend_method_num,
        )

        out = (
            df.merge(std, on=['analysis_datetime', 'run_time'], how='left')
              .sort_values('analysis_datetime')
        )

        # Fill missing smoothed values within each run_time
        out['smoothed'] = (
            out.groupby('run_time', group_keys=False)['smoothed']
            .apply(lambda s: s.interpolate(method='linear', limit_direction='both').ffill().bfill())
        )

        out['normalized_resp'] = out[self.response_type] / out['smoothed']

        return out

    def extract_digits(self, s: str) -> str:
        """Extract digits from the serial portion, ignoring prefix and the last underscore part."""
        # Split on '_' but remove the last part
        parts = s.split('_')
        core = "_".join(parts[:-1])  # everything except the last section
        return ''.join(re.findall(r'\d+', core))


    def compute_rms(self, series: pd.Series, drop_outlier: bool = False) -> tuple[float, int]:
        """
        Compute RMS and number of points from a Series, ignoring NaNs.

        Parameters
        ----------
        series : pd.Series
            Input series to evaluate.
        drop_outlier : bool, default False
            If True, drop the single largest absolute value before computing RMS and n.

        Returns
        -------
        rms : float
            Root-mean-square of the non-NaN values (0.0 if no data).
        n : int
            Number of non-NaN values used.
        """
        s = series.dropna()
        if drop_outlier and len(s) > 0:
            idx = s.abs().idxmax()
            s = s.drop(idx)
        n = len(s)
        if n == 0:
            return 0.0, 0
        rms = np.sqrt((s**2).mean())
        return rms, n

    def sample_diffs(self,
        run_df: pd.DataFrame,
        verbose: bool = True,
        drop_outlier: bool = False,
    ) -> tuple[float, int]:
        """
        Compute sample pair RMS differences (normalized_resp) for flask (run_type_num == 1)
        and tank (run_type_num == 7) data for a single run_time.

        Returns
        -------
        pair_resp_rms : float
            RMS of sample pair response differences.
        pair_resp_n : int
            Number of points used in the sample pair RMS.
        """

        run_time = run_df['run_time'].iloc[0]
        df = run_df.loc[run_df['data_flag_int'] == 0].copy()

        # ---------- Flask pairs (run_type_num == 1) ----------
        flask_df = (
            df.loc[df['run_type_num'] == 1]
            .groupby('sample_id')
            .agg({
                'pair_id_num': 'first',
                'normalized_resp': 'mean',
            })
        )

        if not flask_df.empty:
            flask_df['resp_diff'] = flask_df.groupby('pair_id_num')['normalized_resp'].diff()
        else:
            flask_df['resp_diff'] = pd.Series(dtype='float64')

        # ---------- Tank repeated runs (run_type_num == 7) ----------
        tank_df = (
            df.loc[df['run_type_num'] == 7]
            .groupby('port_info')
            .agg({
                'normalized_resp': 'mean',
            })
        )

        if not tank_df.empty:
            tank_df = tank_df.reset_index()
            tank_df['serial_num'] = tank_df['port_info'].apply(self.extract_digits)
            tank_df['resp_diff']  = tank_df.groupby('serial_num')['normalized_resp'].diff()
        else:
            tank_df = pd.DataFrame(
                columns=['port_info', 'normalized_resp', 'serial_num', 'resp_diff']
            )

        # ---------- Combine flask + tank replicate diffs ----------
        resp_pieces = []
        if not flask_df['resp_diff'].empty:
            resp_pieces.append(flask_df['resp_diff'])
        if not tank_df['resp_diff'].empty:
            resp_pieces.append(tank_df['resp_diff'])

        if resp_pieces:
            combined_resp_diff = pd.concat(resp_pieces, ignore_index=True)
        else:
            combined_resp_diff = pd.Series(dtype='float64')

        pair_resp_rms, pair_resp_n = self.compute_rms(combined_resp_diff, drop_outlier=drop_outlier)

        if verbose:
            print(
                f"{run_time}: pair resp RMS = {pair_resp_rms:0.4f} (n={pair_resp_n})"
            )

        return pair_resp_rms, pair_resp_n

    def find_best_detrend_per_run(self,
        df: pd.DataFrame,
        methods=(1, 2, 3, 4, 5, 6),
        default_method: int = 2,
        margin_frac: float = 0.10,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        For each run_time in df, pick the detrend_method_num that beats the default
        method by at least `margin_frac` on rep_resp_rms. If no method clears the
        margin, stick with the default (Lowess ~5 points).

        Returns a DataFrame with columns: run_time, best_method, best_rms, stats
        """

        # Ensure default is included
        all_methods = sorted(set(methods) | {default_method})

        # 1) Build normalized data once per method
        norm_by_method: dict[int, pd.DataFrame] = {}
        for meth in all_methods:
            if verbose:
                print(f"Building normalized data for method {meth}")
            norm_by_method[meth] = self.merge_smoothed_data(
                df, detrend_method_num=meth
            )

        run_times = sorted(df['run_time'].unique())
        results = []

        # 2) Loop over run_times
        for rt in run_times:
            if verbose:
                print(f"=== run_time {rt} ===")

            best_method = default_method
            best_stats = (np.inf, 0)

            # Evaluate default first
            default_df = norm_by_method[default_method]
            default_run = default_df.loc[default_df['run_time'] == rt].copy()
            if not default_run.empty:
                default_stats = self.sample_diffs(default_run, verbose=False)
                best_stats = default_stats
                if verbose:
                    print(
                        f"  default {default_method}: RMS={default_stats[0]:.6g} (n={default_stats[1]})"
                    )
            else:
                if verbose:
                    print(f"  default {default_method}: no data for this run_time")

            # 3) Try each other method for this run_time
            for meth in all_methods:
                if meth == default_method:
                    continue

                df_norm = norm_by_method[meth]
                run_df = df_norm.loc[df_norm['run_time'] == rt].copy()
                if run_df.empty:
                    if verbose:
                        print(f"  method {meth}: no data for this run_time")
                    continue

                stats = self.sample_diffs(run_df, verbose=False)
                rep_resp_rms, pair_resp_n = stats

                if verbose:
                    print(f"  method {meth}: RMS={rep_resp_rms:.6g} (n={pair_resp_n})")

                # Skip if no valid points
                if pair_resp_n == 0 or np.isnan(rep_resp_rms):
                    continue

                # Enforce margin rule vs current best (starts at default)
                if rep_resp_rms < best_stats[0] * (1 - margin_frac):
                    best_stats = stats
                    best_method = meth

            results.append({
                'run_time': rt,
                'best_method': best_method,
                'best_rms': best_stats[0],
                'stats': best_stats,
            })

        return pd.DataFrame(results)

    def detrend_stats_for_run(self,
        run_df: pd.DataFrame,
        methods=(1, 2, 3, 4, 5, 6),
        default_method: int = 2,
        margin_frac: float = 0.10,
        drop_outlier: bool = False,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, int | None, float]:
        """
        Compute rep_resp_rms for a single run_time across multiple detrend methods.
        Also return the best method using the same margin rule as find_best_detrend_per_run.

        Parameters
        ----------
        run_df : pd.DataFrame
            Dataframe already filtered to a single run_time.
        methods : iterable[int]
            detrend_method_num values to evaluate.
        default_method : int
            Baseline method (Lowess ~5 points) that others must beat by margin_frac.
        margin_frac : float
            Required fractional improvement over the default to switch methods.
        drop_outlier : bool, default False
            Drop the single largest absolute diff before RMS (passed to sample_diffs).
        verbose : bool, default False
            Print per-method RMS.

        Returns
        -------
        stats_df : pd.DataFrame
            Columns: run_time, detrend_method_num, rep_resp_rms, pair_resp_n
        best_method : int | None
            Method chosen via margin rule (or min RMS if default missing); None if no data.
        best_rms : float
            RMS for the selected best_method (NaN if no data).
        """
        
        # use only unflagged data
        run_df = run_df.loc[run_df['data_flag'] == '...'].copy()
         
        if run_df.empty:
            return (
                pd.DataFrame(columns=['run_time', 'detrend_method_num', 'rep_resp_rms', 'pair_resp_n']),
                None,
                np.nan,
            )

        # For BLD1 (inst_num == 220), use ref tank variability instead of sample_diffs
        if 'inst_num' in run_df and int(run_df['inst_num'].iloc[0]) == 220:
            return self.find_best_reftank_norm_resp(
                run_df,
                methods=methods,
                default_method=default_method,
                margin_frac=margin_frac,
                ref_port=11,
                verbose=verbose,
            )

        run_time = run_df['run_time'].iloc[0] if 'run_time' in run_df else None
        results = []
        valid = []  # (meth, rms, n)
        default_stats = None

        all_methods = sorted(set(methods) | {default_method})

        for meth in all_methods:
            df_norm = self.merge_smoothed_data(run_df, detrend_method_num=meth)
            rep_resp_rms, pair_resp_n = self.sample_diffs(
                df_norm,
                verbose=False,
                drop_outlier=drop_outlier,
            )

            if verbose:
                print(f"method {meth}: RMS={rep_resp_rms:.6g} (n={pair_resp_n})")

            results.append({
                'run_time': run_time,
                'detrend_method_num': meth,
                'rep_resp_rms': rep_resp_rms,
                'pair_resp_n': pair_resp_n,
            })

            if pair_resp_n > 0 and not np.isnan(rep_resp_rms):
                valid.append((meth, rep_resp_rms, pair_resp_n))
                if meth == default_method:
                    default_stats = (rep_resp_rms, pair_resp_n)

        # Select best method
        best_method = None
        best_rms = np.nan

        if not valid:
            stats_df = pd.DataFrame(results)
            return stats_df, best_method, best_rms

        if default_stats is not None:
            best_method = default_method
            best_rms = default_stats[0]
            for meth, rms, _ in valid:
                if meth == default_method:
                    continue
                if rms < best_rms * (1 - margin_frac):
                    best_method = meth
                    best_rms = rms
        else:
            # Default missing; pick the lowest RMS
            best_method, best_rms, _ = min(valid, key=lambda x: x[1])

        stats_df = pd.DataFrame(results)
        return stats_df, best_method, best_rms

    def find_best_reftank_norm_resp(
        self,
        df: pd.DataFrame,
        methods=(1, 2, 3, 4, 5, 6),
        default_method: int = 2,
        margin_frac: float = 0.10,
        ref_port: int = 11,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, int | None, float]:
        """
        Find best detrend method for a single run_time using reference tank
        variability (port == ref_port). Point-to-point (method 1) is ignored for
        selection because it trivially returns zero.

        Returns the same tuple shape as detrend_stats_for_run:
        (stats_df, best_method, best_rms)
        where stats_df has columns run_time, detrend_method_num, rep_resp_rms, rep_resp_n
        (rep_resp_rms here is the ref tank std, rep_resp_n is its count).
        """

        if df.empty:
            return (
                pd.DataFrame(columns=['run_time', 'detrend_method_num', 'rep_resp_rms', 'rep_resp_n']),
                None,
                np.nan,
            )
            
        if default_method == 1:
            default_method = 2  # avoid point-to-point as default

        run_time = df['run_time'].iloc[0] if 'run_time' in df else None
        all_methods = sorted(set(methods) | {default_method})

        # Precompute normalized data per method
        norm_by_method: dict[int, pd.DataFrame] = {}
        for meth in all_methods:
            if verbose:
                print(f"Building normalized data for method {meth}")
            norm_by_method[meth] = self.merge_smoothed_data(
                df, detrend_method_num=meth
            )

        def ref_stats(df_norm: pd.DataFrame) -> tuple[float, int]:
            if 'port' in df_norm:
                ref = df_norm.loc[df_norm['port'] == ref_port, 'normalized_resp']
            elif 'port_idx' in df_norm:
                ref = df_norm.loc[df_norm['port_idx'] == ref_port, 'normalized_resp']
            else:
                return np.nan, 0
            ref = ref.dropna()
            if ref.empty:
                return np.nan, 0
            return ref.std(), len(ref)

        results = []
        valid = []  # (meth, std, n)
        default_stats = None

        # Evaluate all methods
        for meth in all_methods:
            df_norm = norm_by_method[meth]
            r_std, r_n = ref_stats(df_norm)

            if verbose:
                print(f"  method {meth}: STD={r_std:.6g} (n={r_n})")

            results.append({
                'run_time': run_time,
                'detrend_method_num': meth,
                'rep_resp_rms': r_std,
                'rep_resp_n': r_n,
            })

            if meth == 1:
                continue  # skip P2P for selection
            if r_n == 0 or np.isnan(r_std):
                continue

            valid.append((meth, r_std, r_n))
            if meth == default_method:
                default_stats = (r_std, r_n)

        # Select best method with margin rule
        best_method = None
        best_rms = np.nan

        if not valid:
            return pd.DataFrame(results), best_method, best_rms

        if default_stats is not None:
            best_method = default_method
            best_rms = default_stats[0]
            for meth, std, _ in valid:
                if meth == default_method:
                    continue
                if std < best_rms * (1 - margin_frac):
                    best_method = meth
                    best_rms = std
        else:
            best_method, best_rms, _ = min(valid, key=lambda x: x[1])

        stats_df = pd.DataFrame(results)
        return stats_df, best_method, best_rms

class M4_Instrument(HATS_DB_Functions):
    """ Class for accessing M4 specific functions in the HATS database. """
    
    RUN_TYPE_MAP = {
        "All": None,        # no filter
        "Flasks": 1,        # run_type_num
        #"Calibrations": 2,
        "PFPs": 5,
    }
    STANDARD_RUN_TYPE = 8
    EXCLUDE = [6, 7]     # run_type_num to exclude from autoscaling (zero air and tank runs)
    
    MARKER_MAP = {
        # run_type_num
        1: 'o',   # Flask
        5: 's',   # PFP
        6: 'v',   # Zero
        7: '^',   # Tank
        8: 'D',   # Standard
    }
    
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
        
        self.norm = Normalizing(self.inst_id, self.STANDARD_RUN_TYPE, 'run_type_num', self.response_type)
                
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
                #AND detrend_method_num != 3
                AND run_time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
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

        df['data_flag_int'] = 0
        df.loc[df['data_flag'] != '...', 'data_flag_int'] = 1
        
        # build a port_idx for plotting colors
        mask = df['run_type_num'].eq(5)     # pfp runtype
        base = pd.to_numeric(df['port'], errors='coerce').astype('float64')
        pfp  = pd.to_numeric(df['flask_port'], errors='coerce').astype('float64') + 20
        res = base.copy()
        res.loc[mask] = pfp.loc[mask]          # explicit assignment avoids where/mask downcast warning
        df['port_idx'] = res.round().astype('Int64')   # final, intentional cast to nullable int
        
        df = self.add_port_labels(df)       # port labels, colors, and markers
        
        return df.sort_values('analysis_datetime')
        
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

        # assign colors to sites
        cmap = plt.get_cmap('tab20')
        site_colors = {site: cmap(i % 20) for i, site in enumerate(self.LOGOS_sites)}

        # Start with site-based colors
        df['port_color'] = df['site'].map(site_colors).fillna('gray')

        # Override when port == 14 → gray
        df.loc[df['run_type_num'] == self.STANDARD_RUN_TYPE, 'port_color'] = 'red'

        # Override when port_info == 'zero_air' → black
        df.loc[df['port_info'] == 'zero_air', 'port_color'] = 'black'  
        
        df['port_marker'] = df['run_type_num'].map(self.MARKER_MAP).fillna('o')

        return df            

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

    def return_preferred_channel(self, gas: str) -> str | None:
        gas_l = gas.lower()
        # override for CFC11/113
        if gas_l in ('cfc11','cfc113'):
            return 'c'
        # otherwise lookup in the precomputed map
        return self.molecule_channel_map.get(gas_l)
    
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
                AND height <> -999
                #AND detrend_method_num != 3
                AND run_time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
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

        df['data_flag_int'] = 0
        df.loc[df['data_flag'] != '...', 'data_flag_int'] = 1
        
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
            
    def load_calcurves(self, pnum, channel, earliest_run):
        """
        Returns the calibration curves from ng_response for a given parameter number and channel.
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

        df['func_index'] = None  # default

        # Condition 2: coef3 == 0 and abs(coef2) > 0 → func_index = 1
        df.loc[(df['coef3'] == 0) & (df['coef2'].abs() > 0), 'func_index'] = 1

        # Condition 3: abs(coef3) > 0 → func_index = 2
        df.loc[df['coef3'].abs() > 0, 'func_index'] = 2

        # Condition 1: coef3 == 0 and coef2 == 0 and abs(coef1) > 0 → func_index = 0
        df.loc[(df['coef3'] == 0) & (df['coef2'] == 0) & (df['coef1'].abs() > 0), 'func_index'] = 0

        return df
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
        self.inst_num = 220
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
                AND height <> -999
                #AND detrend_method_num != 3
                AND run_time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
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

        df['data_flag_int'] = 0
        df.loc[df['data_flag'] != '...', 'data_flag_int'] = 1
        
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
