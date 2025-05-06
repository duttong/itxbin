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
import warnings

from m4_export import M4_base


class M4_Processing(M4_base):
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
        """Load data from the database with date filtering."""
        
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
        query = f"""
            SELECT analysis_datetime, run_time, run_type_num, port_info, area, mole_fraction
            FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                AND area != 0
                AND analysis_datetime BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'])
        df['run_time']          = pd.to_datetime(df['run_time'])
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['area']              = df['area'].astype(float)
        df['mole_fraction']     = df['mole_fraction'].astype(float)
        df['parameter_num']     = pnum
        self.data = df.sort_values('analysis_datetime')
        
    def calculate_normalized_area(self, df, min_pts=8):
        std = (
            df.loc[df['run_type_num'] == self.STANDARD_RUN_TYPE,
                   ['analysis_datetime','run_time','area']]
              .dropna()
              .sort_values('analysis_datetime')
              .copy()
        )
        if len(std) < min_pts:
            std['smoothed'] = np.nan
            return std[['analysis_datetime','run_time','smoothed']]

        std['ts'] = std['analysis_datetime'].view(np.int64) // 10**9
        frac = min(max(min_pts/len(std), 0.3), 1.0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
            std['smoothed'] = (
                std
                .groupby('run_time')
                .apply(lambda seg: pd.Series(
                    lowess(
                        seg['area'].values,
                        seg['ts'].values,
                        frac=frac,
                        return_sorted=False
                    ),
                    index=seg.index
                ))
                .droplevel(0)
            )

        std['smoothed'] = std['smoothed'].interpolate(method='linear',
                                                     limit_direction='both')
        return std[['analysis_datetime','run_time','smoothed']]

    def merge_smoothed_data(self, min_pts=8):
        df = self.data.sort_values('analysis_datetime')
        std = self.calculate_normalized_area(df, min_pts=min_pts)

        out = (
            df
            .merge(std, on=['analysis_datetime','run_time'], how='left')
            .sort_values('analysis_datetime')
        )

        def _fill_run(g):
            g['smoothed'] = (
                g['smoothed']
                 .interpolate(method='linear', limit_direction='both')
                 .fillna(method='ffill')
                 .fillna(method='bfill')
            )
            return g

        out = out.groupby('run_time', group_keys=False).apply(_fill_run)
        out['normalized_area'] = out['area'] / out['smoothed']
        return out


def main():
    parser = argparse.ArgumentParser(
        description="Process M4 data and optionally plot results"
    )
    parser.add_argument(
        '-p', '--parameter-num',
        type=int,
        required=True,
        help="Parameter number"
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
        choices=['yes', 'no'],
        default='yes',
        help="Show figures? yes/no"
    )
    args = parser.parse_args()

    m4 = M4_Processing()
    m4.load_data(
        pnum=args.parameter_num,
        start_date=args.start_date,
        end_date=args.end_date
    )
    df = m4.merge_smoothed_data()

    # get analyte names
    analytes = m4.m4_analytes()
    inv = {int(v): k for k, v in analytes.items()}
    title_text = f"{inv.get(args.parameter_num, 'Unknown')} ({args.parameter_num})"

    colors = df['run_type_num'].map(m4.COLOR_MAP).fillna('gray')
    run_map = {v: k for k, v in m4.run_type_num().items()}
    legend_handles = [
        mpatches.Patch(color=col, label=run_map[rt])
        for rt, col in m4.COLOR_MAP.items()
        if isinstance(rt, int) and rt in run_map
    ]

    if args.figures == 'yes':
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
            label="Normalized area"
        )
        ax2.legend(handles=legend_handles, title="Run Types")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
