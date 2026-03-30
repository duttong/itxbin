#! /usr/bin/env python

import io
import argparse
from contextlib import redirect_stdout
import pandas as pd

from gcwerks_export import GCwerks_export
from bld1_gcwerks2db import main as bld1_gcwerks2db


class BLD1_Process(GCwerks_export):
    """ Class hardcoded for the BLD1 instrument """

    def __init__(self, inst='bld1', prefix='bld1', flagged=True):
        super().__init__(inst, prefix)
        self.flagged = flagged
        suffix = '_flagged' if flagged else ''
        self.results_file = f'/hats/gc/bld1/results/bld1_gcwerks_all{suffix}.csv'

    def export_onefile(self, csv=True, report='/hats/gc/itxbin/report.conf'):
        years = self.gcwerks_years()
        mindate = f'{years[0]}0101'
        maxdate = f'{str(pd.Timestamp.today().year)[2:4]}1231.2359'
        self.gcwerks_export('all', mindate=mindate, maxdate=maxdate, csv=csv, flagged=self.flagged, report=report)

    def export_years(self, mol, start_year=1998, end_year=pd.Timestamp.today().year):
        years = self.gcwerks_years()
        for year in range(int(start_year), int(end_year) + 1):
            if str(year)[2:4] in years:
                mindate = f'{str(year)[2:4]}0101'
                maxdate = f'{str(year)[2:4]}1231.2359'
                self.gcwerks_export(mol, mindate, maxdate, flagged=self.flagged)

    def gcwerks_export(self, mol, mindate=False, maxdate=False, csv=True, flagged=False, mk2yrfile=False, report='/hats/gc/itxbin/report.conf'):
        if mol == 'all' and csv and flagged:
            results_file = self.export_path / f'{self.prefix}_gcwerks_all_flagged.csv'
            print(f'Exporting {mol} to {results_file}')
            default_results = self.export_path / f'{self.prefix}_gcwerks_all.csv'
            with redirect_stdout(io.StringIO()):
                exported = super().gcwerks_export(
                    mol,
                    mindate=mindate,
                    maxdate=maxdate,
                    csv=csv,
                    flagged=flagged,
                    mk2yrfile=mk2yrfile,
                    report=report,
                )
            if exported and default_results.exists():
                default_results.rename(results_file)
            return exported

        return super().gcwerks_export(
            mol,
            mindate=mindate,
            maxdate=maxdate,
            csv=csv,
            flagged=flagged,
            mk2yrfile=mk2yrfile,
            report=report,
        )

    def read_results(self, year='all'):
        df = pd.read_csv(self.results_file, skipinitialspace=True, parse_dates=[0])
        df.set_index(df.time, inplace=True)
        if year == 'all':
            return df
        else:
            return df[str(year)]


if __name__ == '__main__':

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
    opt.add_argument('--flagged', action=argparse.BooleanOptionalAction, default=True,
        help='Use flagged GCwerks export data (default: enabled).')

    options = opt.parse_args()
    bld1 = BLD1_Process(flagged=options.flagged)

    reportfile = '/hats/gc/bld1/results/bld1_report.conf'

    if options.year:
        bld1.export_years('all', start_year=options.year, end_year=options.year)
        bld1_gcwerks2db(year=options.year, flagged=options.flagged)
        quit()

    # bld1.export_years(options.mol)
    bld1.export_onefile(report=reportfile)
    
    # upload gcwerks results of current year to DB (added 20251105)
    current_year = bld1.read_results(year='all').index[-1].year
    bld1_gcwerks2db(year=current_year, flagged=options.flagged)
