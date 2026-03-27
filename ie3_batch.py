#!/usr/bin/env python3

import argparse
import time

import pandas as pd

from logos_instruments import IE3_Instrument


class IE3_batch(IE3_Instrument):

    def __init__(self, site="smo"):
        super().__init__(site=site)
        self.t0 = time.time()

    def update_runs(self, pnum, channel=None, start_date=None, end_date=None, verbose=False):
        """Calculate mole fractions for a pnum/channel and return the DataFrame."""
        df = self.load_data(
            pnum=pnum,
            channel=channel,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
        )
        if df.empty:
            return pd.DataFrame()

        if verbose:
            print(f"Loaded {len(df)} rows. elapsed: {time.time() - self.t0:.2f}s")

        df = self.calc_mole_fraction(df)

        if verbose:
            print(f"Mole fractions calculated. elapsed: {time.time() - self.t0:.2f}s")

        return df

    def main(self):
        parser = argparse.ArgumentParser(
            description="Recalculate and optionally insert IE3 mole fractions."
        )
        parser.add_argument(
            '-p', '--parameter-num',
            type=str,
            required=True,
            help="Parameter number or 'all' to process all analytes.",
        )
        parser.add_argument(
            '-c', '--channel',
            type=str,
            help="Channel (a, b, c, …). Required when -p is not 'all'.",
        )
        parser.add_argument(
            '-s', '--start-date',
            type=str,
            help="Start date in YYMM format (e.g. '2503'). Defaults to last 30 days.",
        )
        parser.add_argument(
            '-e', '--end-date',
            type=str,
            help="End date in YYMM format (e.g. '2505').",
        )
        parser.add_argument(
            '--site',
            type=str,
            default='smo',
            help="Station code (default: smo).",
        )
        parser.add_argument(
            '-i', '--insert',
            action='store_true',
            help="Write recalculated mole fractions back to the database.",
        )
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
        )
        args = parser.parse_args()

        if args.parameter_num.lower() == 'all':
            sql = (
                "SELECT param_num, channel, display_name "
                f"FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
            )
            adf = pd.DataFrame(self.doquery(sql))
            for row in adf.itertuples(index=False):
                pnum = int(row.param_num)
                ch = row.channel
                print(f"Processing {row.display_name} (pnum={pnum} ch={ch})")
                df = self.update_runs(
                    pnum, channel=ch,
                    start_date=args.start_date, end_date=args.end_date,
                    verbose=args.verbose,
                )
                if args.insert:
                    self.upsert_mole_fractions(df)
            print(f"All analytes done. Total time: {time.time() - self.t0:.2f}s")
        else:
            pnum = int(args.parameter_num)
            df = self.update_runs(
                pnum, channel=args.channel,
                start_date=args.start_date, end_date=args.end_date,
                verbose=args.verbose,
            )
            print(df[['analysis_time', 'port', 'normalized_resp', 'mole_fraction']].to_string())
            if args.insert:
                self.upsert_mole_fractions(df)
            print(f"Done ({len(df)} rows). Total time: {time.time() - self.t0:.2f}s")


if __name__ == '__main__':
    import sys
    # Pull --site before argparse so we can pass it to __init__
    site = 'smo'
    for i, arg in enumerate(sys.argv):
        if arg == '--site' and i + 1 < len(sys.argv):
            site = sys.argv[i + 1]
    ie3 = IE3_batch(site=site)
    ie3.main()
