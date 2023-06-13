#! /usr/bin/env python

import argparse
import pandas as pd

from gcwerks_export import GCwerks_export


class FE3_Process(GCwerks_export):
    """ Class hardcoded for the FE3 instrument """

    def __init__(self, inst='fe3', prefix='fe3'):
        super().__init__(inst, prefix)
        self.results_file = f'/hats/gc/{inst}/results/fe3_gcwerks_all.csv'

    def read_results(self, year='all'):
        df = pd.read_csv(self.results_file, skipinitialspace=True, parse_dates=[0])
        df.set_index(df.time, inplace=True)
        if year == 'all':
            return df
        else:
            return df[str(year)]


if __name__ == '__main__':

    fe3 = FE3_Process()

    opt = argparse.ArgumentParser(
        description='Exports FE3 data from GCwerks.'
    )
    opt.add_argument('mol', nargs='?', default='all',
        help=f'Select a single molecule to export or default \
        to "all".  Valid mol variables: {sorted(fe3.gcwerks_peaks())}')
    opt.add_argument('-year', action='store', type=int,
        help="Export this year's data.")

    options = opt.parse_args()

    if options.year:
        fe3.export_years(options.mol, start_year=options.year, end_year=options.year)
        quit()

    # fe3.export_years(options.mol)
    fe3.export_onefile()
