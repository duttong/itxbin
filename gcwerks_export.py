#! /home/hats/gdutton/anaconda3/bin/python
""" Python 3.6 updates 181029 """

import shlex
from datetime import date
import pandas as pd
from pathlib import Path
from subprocess import run, Popen, DEVNULL, PIPE
import multiprocessing as mp
from os import chmod
import time

today = date.today()


class GCwerks_export:

    def __init__(self, inst, prefix, export_path='default'):
        self.inst = inst        # instrument code used in gcdir path
        self.prefix = prefix    # prefix for exported data file name
        self.gcdir = f'/hats/gc/{self.inst}'
        self.peaklist = self.gcwerks_peaks()
        self.repmols = {}       # dict of replacement names for mols.
        if export_path == 'default':
            self.export_path = Path(f'{self.gcdir}/results')
        else:
            self.export_path = Path(export_path)

    def gcwerks_peaks(self):
        """ Returns a list of peaks integrated by GCwerks. The routine reads
            the GCwerks config file called peak.list """
        peaklist = Path(f'{self.gcdir}/config/peak.list')
        peaks_df = pd.read_csv(peaklist, sep='\\s+', comment='#', usecols=[0, 1], names=['mol', 'chan'])
        return list(peaks_df[peaks_df.columns[0]])

    def gcwerks_years(self):
        """ List of year directories in self.gcdir path chronologically sorted. """
        yy = [int(yy.name) for yy in Path(self.gcdir).glob('??')]
        yyyy = sorted([y+2000 if y < 95 else y+1900 for y in yy])
        return [str(y)[2:4] for y in yyyy]

    def valid_mol(self, mol):
        """ Returns true if the mol name is in the GCwerks DB and can be
            exported. """
        if mol not in self.peaklist:
            if mol != 'all':
                # print(f'Cannot export {mol}. Missing from peak.list file for {self.inst}')
                return False
        return True

    def gcwerks_export(self, mol, mindate=False, maxdate=False, csv=True, mk2yrfile=False, report='/hats/gc/itxbin/report.conf'):
        """ Exports GCwerks data to a .csv file. Set mol="all" for one file
            with all of the molecules listed in peaks.list
            mindate and maxdate can be of the form YYMMDD 
            use report for a custom output report format (added 210901)
        """

        if not self.valid_mol(mol):
            return False

        # Determine results_file name
        if mol == 'all':
            results_file = self.export_path / f'{self.prefix}_all.txt'
            if csv:
                results_file = self.export_path / f'{self.prefix}_gcwerks_all.csv'
        else:
            results_file = self.export_path / f'{self.prefix}_{self.short_mol_name(mol)}.csv'
        print(f'Exporting {mol} to {results_file}')

        # create temporary report_peak file for gcexport to use
        if mol == 'all':
            report_peak = Path('/tmp/peak_list.txt')
            with open(report_peak, 'w') as f:
                for m in self.peaklist:
                    f.write(f'{m}\n')
        else:
            report_peak = Path(f'/tmp/peak_{mol}.txt')
            with open(report_peak, 'w') as f:
                f.write(mol)
        chmod(report_peak, 0o0664)

        # create gcexport command and send results to a file
        gcexport = Path('/hats/gc/gcwerks-3/bin/gcexport')
        cmd = f'{gcexport} -gcdir {self.gcdir} -peaklist {report_peak} '
        cmd += f'-format {report} '
        cmd += '-missingvalue -999 '
        if mindate is not False:
            cmd += f'-mindate {mindate} '
        if maxdate is not False:
            cmd += f'-maxdate {maxdate} '
        if csv:
            cmd += ' -csv'

        with open(results_file, 'w') as f:
            gcw = Popen(shlex.split(cmd), stdout=PIPE, stderr=DEVNULL)
            run('uniq', stdin=gcw.stdout, stdout=f)
        
        try:
            chmod(results_file, 0o0664)
        except PermissionError:
            pass

        if mk2yrfile:
            self.create_2year_file(results_file)

        # cleanup
        report_peak.unlink()

        return True

    def create_2year_file(self, results_file):
        """ The format of the GCwerks csv file has a column called 'time',
            that is Date followed by Time (ie 2020-07-27 21:33:00). Igor Pro
            does not like the space between the two fields. Reformatted below. """
        outfile = f'{results_file.parent}/{results_file.stem}_2yr.csv'
        df = pd.read_csv(results_file, parse_dates=True, skipinitialspace=True, index_col='time')

        # save two years of data
        if not df.empty:
            end = df.index.max()
            start = end - pd.DateOffset(months=24)
            df = df.loc[df.index >= start]
        print(f'Creating {outfile}')
        df.to_csv(outfile, index=True)
        chmod(outfile, 0o0664)

    def export_onefile(self, csv=True, report='/hats/gc/itxbin/report.conf'):
        """ Exports all data (all mols for all years) to a single file. """
        years = self.gcwerks_years()
        mindate = f'{years[0]}0101'
        maxdate = f'{str(today.year)[2:4]}1231.2359'
        self.gcwerks_export('all', mindate=mindate, maxdate=maxdate, csv=csv, report=report)

    def short_mol_name(self, mol):
        """ Returns the replacement name for a molecule used in
            the file name. """
        if self.repmols is None:
            return mol
        try:
            shtmol = self.repmols[mol.lower()]
        except KeyError:
            shtmol = mol.upper()
        return shtmol

    def export_years(self, mol, start_year=1998, end_year=today.year):
        """ Export data into yearly files. Use 4-digit year. """

        years = self.gcwerks_years()
        for year in range(int(start_year), int(end_year)+1):
            if str(year)[2:4] in years:
                # results_file = self.export_path / f'{self.prefix}_{self.short_mol_name(mol)}_{year}.csv'
                mindate = f'{str(year)[2:4]}0101'
                maxdate = f'{str(year)[2:4]}1231.2359'
                self.gcwerks_export(mol, mindate, maxdate)

    def export_mols(self, mol_list=None, csv=True, mk2yrfile=False):
        """ Exports a list of molecules, by default the list is from peak.list
            file in the gcwerks config directory. """

        mols = self.peaklist if mol_list is None else mol_list
        years = self.gcwerks_years()
        mindate = f'{years[0]}0101'
        maxdate = f'{str(today.year)[2:4]}1231.2359'
        t0 = time.time()

        pool = mp.Pool(7)
        for mol in mols:
            pool.apply_async(self.gcwerks_export, args=([mol, mindate, maxdate, csv, mk2yrfile]))
        pool.close()
        pool.join()

        print(f'Elapsed time: {time.time()-t0:.3f} s')
