#! /usr/bin/env python

import io
import argparse
from contextlib import redirect_stdout
import pandas as pd

from gcwerks_export import GCwerks_export


class FE3_Process(GCwerks_export):
    """ Class hardcoded for the FE3 instrument """

    def __init__(self, inst='fe3', prefix='fe3', flagged=False):
        super().__init__(inst, prefix)
        self.flagged = flagged
        suffix = '_flagged' if flagged else ''
        self.results_file = f'/hats/gc/{inst}/results/fe3_gcwerks_all{suffix}.csv'

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

    fe3 = FE3_Process()

    opt = argparse.ArgumentParser(
        description='Exports FE3 data from GCwerks.'
    )
    opt.add_argument('mol', nargs='?', default='all',
        help=f'Select a single molecule to export or default \
        to "all".  Valid mol variables: {sorted(fe3.gcwerks_peaks())}')
    opt.add_argument('-year', action='store', type=int,
        help="Export this year's data.")
    opt.add_argument('--flagged', action='store_true',
        help='Include flagged data and write fe3_gcwerks_all_flagged.csv.')

    options = opt.parse_args()
    fe3 = FE3_Process(flagged=options.flagged)

    if options.year:
        fe3.export_years(options.mol, start_year=options.year, end_year=options.year)
        quit()

    # fe3.export_years(options.mol)
    fe3.export_onefile()
