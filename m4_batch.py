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

    def calculate_mole_fraction(self, df):
        """
        Compute mole_fraction = (a0 + a1·days_elapsed) * x
        where days_elapsed is days since 1900-01-01 relative to run_time,
        and x is normalized_resp.
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

            mf.loc[grp.index] = (a0 + a1 * days) * grp['normalized_resp']

        out = df.copy()
        out['mole_fraction'] = mf
        return out

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
            df = m4.load_data(
                pnum=pnum,
                start_date=args.start_date,
                end_date=args.end_date
            )
            if m4.data.empty:
                continue
            
            df = m4.calculate_mole_fraction(df)

            if args.insert:
                m4.upsert_mole_fractions(df)

        # No figures when processing all analytes
        print(f"Processing complete for all analytes. Total time: {time.time() - t0:.2f} seconds")
        return
    else:
        # Process a single parameter
        pnum = int(args.parameter_num)
        df = m4.load_data(
            pnum=pnum,
            start_date=args.start_date,
            end_date=args.end_date
        )
        if m4.data.empty:
            return

        df = m4.calculate_mole_fraction(df)

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
            m4.upsert_mole_fractions(df)
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
                df['mole_fraction'],
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
