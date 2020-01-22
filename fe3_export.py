#! /home/hats/gdutton/anaconda3/bin/python

import pandas as pd
from pathlib import Path
from subprocess import run, DEVNULL
import shlex
import argparse
from os import chmod
from datetime import date

today = date.today()


class GCwerks_export:

    def __init__(self, inst, prefix, break_year=True):
        self.inst = inst        # instrument code used in gcdir path
        self.prefix = prefix    # prefix for exported data file name
        self.break_year = break_year
        self.gcdir = f'/hats/gc/{self.inst}'
        self.export_path = Path(f'{self.gcdir}/results')
        self.report_peak = '/tmp/peak.list'
        self.mols = self.gcwerks_peaks()

    def gcwerks_peaks(self):
        """ Returns a list of peaks integrated by GCwerks. The routine reads
            the GCwerks config file called peak.list """
        peaklist = Path(f'{self.gcdir}/config/peak.list')
        peaks_df = pd.read_csv(peaklist, sep='\\s+', comment='#', names=['mols','ch','fullname'])
        return list(peaks_df.mols)

    def export_peaklist(self, mol):
        """ Creates /tmp/peak.list file for gcexport
            set mol=='all' to export all gcwerks peaks to a single csv file. """

        if mol.lower() == 'all':
            with open(self.report_peak, 'w') as f:
                for m in self.mols:
                    f.write(f'{m}\n')
        else:
            with open(self.report_peak, 'w') as f:
                f.write(mol)

    def valid_mol(self, mol):
        if mol not in self.mols:
            if mol != 'all':
                print(f'Cannot export {mol}. Missing from peak.list file for {self.inst}')
                quit()

    def gcwerks_export(self, mol, mindate, maxdate, results_file):
        """ exports GCwerks data to a .csv file. Set mol="all" for one file
            with all of the molecules listed in peaks.list """

        self.valid_mol(mol)
        self.export_peaklist(mol)

        # setup paths and gcexport command
        gcexport = Path('/hats/gc/gcwerks-3/bin/gcexport')
        format = '/hats/gc/itxbin/report.conf'

        # call gcexport and send results to a file
        cmd = f'{gcexport} -gcdir {self.gcdir} -peaklist {self.report_peak} -format {format} '
        cmd += '-missingvalue -999 -csv'
        cmd += f'-mindate {mindate} -maxdate {maxdate}'

        with open(results_file, 'w') as f:
            run(shlex.split(cmd), stdout=f, stderr=DEVNULL)
        chmod(results_file, 0o0664)

    def export_years(self, mol, start_year, end_year=today.year):
        """ Export data into yearly files """

        for year in range(int(start_year), int(end_year)+1):
            results_file = self.export_path / f'{self.prefix}_{mol}_{year}.txt'
            mindate = f'{str(year)[2:4]}0101'
            maxdate = f'{str(year)[2:4]}1231.2359'
            self.gcwerks_export(mol, mindate, maxdate, results_file)

    def export_mols(self, mol_list=None):
        """ exports a list of molecules, by default the list is from peak.list
            file in the gcwerks config directory. """

        mols = self.mols if mol_list is None else mol_list
        for mol in mols:
            self.gcwerks_export(mol)


class FE3_Process(GCwerks_export):
    """ Class hardcoded for the FE3 instrument """

    def __init__(self, inst='agc1', prefix='FE3'):
        super().__init__(inst, prefix)


if __name__ == '__main__':

    fe3 = FE3_Process()

    opt = argparse.ArgumentParser(
        description='Exports FE3 data from GCwerks.'
    )
    opt.add_argument('mol', nargs='?', default='all',
        help=f'Select a single molecule to export or default \
        to "all".  Valid mol variables: {sorted(fe3.mols)}')
    opt.add_argument('-year', action='store', type=int,
        help=f"Export this year's data.")

    options = opt.parse_args()

    if options.year:
        fe3.export_years(options.mol, start_year=options.year, end_year=options.year)
        quit()

    fe3.export_years(options.mol, 2019)
