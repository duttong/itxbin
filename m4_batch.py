#!/usr/bin/env python

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime, timedelta
import time

from logos_instruments import M4_Instrument

class M4_Processing(M4_Instrument):
    """Class for processing M4 data."""
    
    STANDARD_RUN_TYPE = 8
    
    COLOR_MAP = {
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
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.status_label = None
    
    def load_data(self, pnum, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = datetime.strptime(end_date, "%y%m")

        if start_date is None:
            start_date = end_date - timedelta(days=60)
        else:
            start_date = datetime.strptime(start_date, "%y%m")

        start_date_str = start_date.strftime("%Y-%m-01")
        end_date_str = end_date.strftime("%Y-%m-%d")

        print(f"Loading data from {start_date_str} to {end_date_str} for parameter {pnum}")
        # todo: use flags - using low_flow flag
        query = f"""
            SELECT analysis_datetime, run_time, run_type_num, port_info, detrend_method_num, 
                area, mole_fraction, net_pressure, flag, sample_id, pair_id_num
            FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                AND area != 0
                AND detrend_method_num != 3
                AND low_flow != 1
                AND run_time BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            print(f"No data found for parameter {pnum} in the specified date range.")
            self.data = pd.DataFrame()
            return
        
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'])
        df['run_time']          = pd.to_datetime(df['run_time'])
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].astype(int)
        df['area']              = df['area'].astype(float)
        df['net_pressure']      = df['net_pressure'].astype(float)
        df['area']              = df['area']/df['net_pressure']
        df['mole_fraction']     = df['mole_fraction'].astype(float)
        df['parameter_num']     = pnum
        self.data = df.sort_values('analysis_datetime')
        return self.data

    def _smooth_segment(self, seg, frac):
        # some of these filters are not needed.
        # 1) drop bad rows
        #seg = seg.dropna(subset=['ts','area']).sort_values('ts')
        # 2) consolidate duplicates
        #seg = seg.groupby('ts', as_index=False)['area'].mean()
        # 3) skip tiny segments
        if len(seg) < 3 or seg['ts'].max() == seg['ts'].min():
            return pd.Series(seg['area'].values, index=seg.index)
        # 4) do LOWESS
        return pd.Series(
            lowess(seg['area'], seg['ts'], frac=frac, return_sorted=False),
            index=seg.index
        )
        
    def calculate_smoothed_std(self, df, min_pts=8, frac=0.5):
        """ Calculate smoothed standard deviation for the standard run type.
            This function uses LOWESS smoothing on the area data.
            min_pts is the minimum number of points required to perform smoothing.
            frac is the fraction of points used for smoothing.
            The smoothed values are returned in a new column 'smoothed'.
        """
        std = (
            df.loc[df['run_type_num'] == self.STANDARD_RUN_TYPE,
                    ['analysis_datetime','run_time','detrend_method_num','area']]
                .dropna()
                .sort_values('analysis_datetime')
                .copy()
        )
        
        # Not enough points to smooth (need at least 15 points)
        if len(std) < min_pts*3:
            std['smoothed'] = np.nan
            return std[['analysis_datetime','run_time','smoothed']]

        # keep only those rows *after* the first in each run_time
        # this is to avoid smoothing the first point in each run_time which is often an outlier
        std = std[std.groupby('run_time').cumcount() > 0].copy()
        
        std['ts'] = std['analysis_datetime'].astype(np.int64) // 10**9
        
        detrend_method = std['detrend_method_num'].iat[0]

        if detrend_method == 1:
            # point to point is the same as a small frac for LOWESS
            frac = 0.01

        std['smoothed'] = (
            std
            .groupby('run_time', group_keys=False)[['ts','area']]
            .apply(lambda seg: self._smooth_segment(seg, frac))
        )

        return std[['analysis_datetime','run_time','smoothed']]
    
    def calculate_mole_fraction(self, df):
        """
        Compute mole_fraction = (a0 + a1·days_elapsed) * x
        where days_elapsed is days since 1900-01-01 relative to run_time,
        and x is normalized_area.
        """
        pnum     = df['parameter_num'].iat[0]
        baseline = pd.Timestamp("1900-01-01")
        mf       = pd.Series(index=df.index, dtype=float)

        # cache for scale values keyed by ref_tank
        scale_cache = {}

        for rt, grp in df.groupby('run_time'):
            mask = grp['run_type_num'] == self.STANDARD_RUN_TYPE
            if not mask.any():
                mf.loc[grp.index] = np.nan
                continue

            ref_tank = grp.loc[mask, 'port_info'].iat[0]

            # only call m4_scale_values once per tank
            if ref_tank not in scale_cache:
                scale_cache[ref_tank] = self.scale_values(ref_tank, pnum)
            coefs = scale_cache[ref_tank]

            if coefs is None:
                mf.loc[grp.index] = np.nan
                continue

            a0 = float(coefs['coef0'])
            a1 = float(coefs['coef1'])
            days = (pd.to_datetime(rt) - baseline).days

            mf.loc[grp.index] = (a0 + a1 * days) * grp['normalized_area']

        out = df.copy()
        out['mole_fraction'] = mf
        return out

    def merge_smoothed_data(self):
        df = self.data
        # smoothed std or reference tank injection
        std = self.calculate_smoothed_std(df, min_pts=5, frac=0.5)

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

        out['normalized_area'] = out['area'] / out['smoothed']
        out = self.calculate_mole_fraction(out)
                
        return out
    
    def insert_mole_fractions(self, df):
        """
        Inserts or updates rows in hats.ng_mole_fractions using a batch upsert.
        Any non‑numeric mole_fraction (including blank strings) becomes NULL.
        """
        sql_insert = """
            INSERT INTO hats.ng_mole_fractions (
                analysis_num,
                parameter_num,
                mole_fraction
            ) VALUES (
                %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                mole_fraction = VALUES(mole_fraction)
        """

        # Coerce everything to float, invalid parses → NaN, then round
        df = df.copy()
        df['mole_fraction'] = (
            pd.to_numeric(df['mole_fraction'], errors='coerce')
            .round(5)
        )

        params = []
        for _, row in df.iterrows():
            # Convert pandas NaN → Python None so INSERT writes a NULL
            mf = row.mole_fraction
            mole_fraction = None if pd.isna(mf) else float(mf)

            params.append((
                row.analysis_num,
                row.parameter_num,
                mole_fraction
            ))

            # flush batch if doMultiInsert returns True
            if self.db.doMultiInsert(sql_insert, params):
                params = []

        # any trailing rows
        if params:
            self.db.doMultiInsert(sql_insert, params, all=True)

def main():
    parser = argparse.ArgumentParser(
        description="Process M4 data and optionally plot results"
    )
    parser.add_argument(
        '-p', '--parameter-num',
        type=str,  # Change to str to allow "all"
        required=True,
        help="Parameter number or 'all' to process all analytes"
    )
    parser.add_argument(
        '-s', '--start-date',
        type=str,
        help="Start date in YYMM format (e.g. '2503')"
    )
    parser.add_argument(
        '-e', '--end-date',
        type=str,
        help="End date in YYMM format (e.g. '2505')"
    )
    parser.add_argument(
        '-f', '--figures',
        action='store_true',
        help="Show figures if provided, otherwise no figures"
    )
    parser.add_argument(
        '-i', '--insert',
        action='store_true',
        help="Insert mole fractions into the database if provided"
    )
    args = parser.parse_args()

    m4 = M4_Processing()
    
    t0 = time.time()

    if args.parameter_num.lower() == "all":
        # Process all analytes
        for analyte_name, pnum in m4.analytes.items():
            print(f"Processing analyte: {analyte_name} (Parameter {pnum})")
            m4.load_data(
                pnum=pnum,
                start_date=args.start_date,
                end_date=args.end_date
            )
            if m4.data.empty:
                continue

            df = m4.merge_smoothed_data()

            if args.insert:
                df = m4.return_analysis_nums(df, 'analysis_datetime')
                m4.insert_mole_fractions(df)

        # No figures when processing all analytes
        print(f"Processing complete for all analytes. Total time: {time.time() - t0:.2f} seconds")
        return
    else:
        # Process a single parameter
        pnum = int(args.parameter_num)
        m4.load_data(
            pnum=pnum,
            start_date=args.start_date,
            end_date=args.end_date
        )
        if m4.data.empty:
            return

        df = m4.merge_smoothed_data()

        # get analyte names
        analytes = m4.analytes
        inv = {int(v): k for k, v in analytes.items()}
        title_text = f"{inv.get(pnum, 'Unknown')} ({pnum})"

        colors = df['run_type_num'].map(m4.COLOR_MAP).fillna('gray')
        run_map = {v: k for k, v in m4.run_type_num().items()}
        legend_handles = [
            mpatches.Patch(color=col, label=run_map[rt])
            for rt, col in m4.COLOR_MAP.items()
            if isinstance(rt, int) and rt in run_map
        ]

        if args.insert:
            df = m4.return_analysis_nums(df, 'analysis_datetime')
            m4.insert_mole_fractions(df)
            print(f"Inserted mole fractions for parameter {pnum} into the database. Total time: {time.time() - t0:.2f} seconds")

        if args.figures:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 12))
            fig.suptitle(title_text, fontsize=16)

            ax1.set_title("Area vs Analysis DateTime")
            ax1.set_xlabel("Analysis DateTime")
            ax1.set_ylabel("Area")
            ax1.xaxis.set_tick_params(rotation=45)
            ax1.grid(True)
            ax1.scatter(
                df['analysis_datetime'],
                df['area'],
                c=colors,
                marker='o', linewidths=0, alpha=0.8,
                label="Raw area"
            )
            color_cycle = ['red', 'green']
            for i, (_, grp) in enumerate(df.groupby('run_time')):
                ax1.plot(
                    grp['analysis_datetime'],
                    grp['smoothed'],
                    color=color_cycle[i % 2],
                    linewidth=1
                )
            ax1.legend(handles=legend_handles, title="Run Types")

            ax2.set_title("Normalized Area")
            ax2.set_xlabel("Analysis DateTime")
            ax2.set_ylabel("Normalized Area")
            ax2.grid(True)
            ax2.scatter(
                df['analysis_datetime'],
                df['normalized_area'],
                c=colors,
                marker='o', linewidths=0, alpha=0.8,
                label="Mole Fraction"
            )
            ax2.legend(handles=legend_handles, title="Run Types")

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        else:
            #print(df.tail(10).to_string(index=False))
            pass


if __name__ == '__main__':
    main()
