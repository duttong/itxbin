#! /usr/bin/env python

import argparse
import pandas as pd

from gcwerks_export import GCwerks_export
from bld1_gcwerks2db import main as bld1_gcwerks2db


class BLD1_Process(GCwerks_export):
    """ Class hardcoded for the BLD1 instrument """

    def __init__(self, inst='bld1', prefix='bld1'):
        super().__init__(inst, prefix)
        self.results_file = '/hats/gc/bld1/results/bld1_gcwerks_all.csv'

    def read_results(self, year='all'):
        df = pd.read_csv(self.results_file, skipinitialspace=True, parse_dates=[0])
        df.set_index(df.time, inplace=True)
        if year == 'all':
            return df
        else:
            return df[str(year)]


if __name__ == '__main__':

    bld1 = BLD1_Process()

    opt = argparse.ArgumentParser(
        description='Exports BLD1 data from GCwerks.'
    )
    """
    opt.add_argument('mol', nargs='?', default='all',
        help=f'Select a single molecule to export or default \
        to "all".  Valid mol variables: {sorted(bld1.gcwerks_peaks())}')
    """
    opt.add_argument('-year', action='store', type=int,
        help="Export this year's data.")

    options = opt.parse_args()

    reportfile = '/hats/gc/bld1/results/bld1_report.conf'

    if options.year:
        bld1.export_years(options.mol, start_year=options.year, end_year=options.year, report=reportfile)
        bld1_gcwerks2db(year=options.year)
        quit()

    # bld1.export_years(options.mol)
    bld1.export_onefile(report=reportfile)
    
    # upload gcwerks results of current year to DB (added 20251105)
    current_year = bld1.read_results(year='all').index[-1].year
    bld1_gcwerks2db(year=current_year)
