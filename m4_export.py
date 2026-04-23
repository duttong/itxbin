#! /usr/bin/env python

import subprocess
import argparse
import sys
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from logos_instruments import M4_Instrument

class M4_GCwerks_Export(M4_Instrument):

    # Map display name -> actual peak name in GCwerks when they differ.
    # Renaming the peak in GCwerks would require rebuilding the entire
    # integration results, so we alias at export time instead.
    GCWERKS_PEAK_ALIAS = {'CFC-11': 'CFC-11b'}

    def __init__(self):
        super().__init__()

    def export_gc_data(self, start_date, molecules, progress=None, flagged=False):
        """
        Calls the gcexport program for each molecule via the subprocess.Popen function. gcexport
        will extract data from gcwerks and send it to a .csv file.

        start_date should be in the form YYMM
        """

        processes = []
        for molecule in molecules:
            if molecule in self.molecules:
                peak = self.GCWERKS_PEAK_ALIAS.get(molecule, molecule)
                suffix = "_flagged" if flagged else ""
                filename = f"data_{molecule}{suffix}.csv"
                params = f"time runtype tank stdtank port psamp {peak}.area {peak}.ht {peak}.rt {peak}.w"
                # params_extra = f"{params} {peak}.skew {peak}.rl.a {peak}.rl.ht {peak}.rl.report {peak}.c.a {peak}.c.ht {peak}.c.report"
                flags_arg = " -flags" if flagged else ""
                command = f"{self.gcexport_path} {self.gc_dir} -csv -nonan{flags_arg} -mindate {start_date} {params} > {self.export_dir}/{filename}"

                # Execute the command and redirect output to /dev/null
                process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                processes.append((process, molecule, peak))
            else:
                print(f'Wrong molecule name: {molecule}')

        for process, molecule, peak in tqdm(processes):
            if progress:
                progress(10)
            process.wait()
            if peak != molecule:
                self._rewrite_header(molecule, peak, flagged=flagged)

    def _rewrite_header(self, molecule, peak, flagged=False):
        """Rewrite the CSV header so `{peak}_*` columns match the display name.

        m4_gcwerks2db.load_gcwerks strips the `{molecule}_` prefix from columns
        using the display name, so an aliased peak name must be renamed on disk.
        """
        suffix = "_flagged" if flagged else ""
        path = self.export_dir / f"data_{molecule}{suffix}.csv"
        if not path.exists():
            return
        with open(path, 'r') as f:
            header = f.readline()
            rest = f.read()
        header = header.replace(f"{peak}_", f"{molecule}_")
        with open(path, 'w') as f:
            f.write(header)
            f.write(rest)

    @staticmethod
    def verify_start_date(date_str):
        """Verify and convert start date to YYMM format."""
        if re.match(r'^\d{8}$', date_str):
            # Convert YYYYMMDD to YYMM
            return date_str[2:6]
        elif re.match(r'^\d{4}$', date_str):
            # Already in YYMM format
            return date_str
        else:
            raise ValueError(f"Invalid date format: {date_str}. Expected format is YYMM or YYYYMMDD.")            

    @staticmethod
    def parse_molecules(molecules):
        if molecules:
            try:
                molecules = molecules.replace(' ', '')   # remove spaces
                return molecules.split(',')
            except AttributeError:      # already a list. just return
                return molecules
        return []

    @staticmethod
    def main():
        m4_export = M4_GCwerks_Export()
        default_molecules = list(m4_export.molecules)
        
        parser = argparse.ArgumentParser(description='Export Perseus data with specified start date.')
        parser.add_argument('start_date', type=str, nargs='?', default=m4_export.start_date,
                            help=f'Start date in the format YYMM (default: {m4_export.start_date}). ')
        parser.add_argument('-m', '--molecules', type=str, default=default_molecules,
                            help='Comma-separated list of molecules. Add quotes around the list if spaces are used. Default all molecules.')
        parser.add_argument('--flagged', action='store_true',
                            help='Include flagged GCwerks data and write *_flagged.csv files.')
        parser.add_argument('--list', action='store_true', help='List all available molecule names.')

        args = parser.parse_args()
        if args.list:
            print(f"Valid molecule names: {default_molecules}")
            quit()

        molecules = m4_export.parse_molecules(args.molecules)     # returns a list of molecules
        start_date = m4_export.verify_start_date(args.start_date)
        m4_export.export_gc_data(start_date, molecules, flagged=args.flagged)


if __name__ == '__main__':
    M4_GCwerks_Export.main()
