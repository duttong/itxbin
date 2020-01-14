#! /home/hats/gdutton/anaconda3/bin/python

import argparse
from datetime import date
from pathlib import Path
from subprocess import run, call
import multiprocessing as mp
import os
import shutil
import gzip

from itx_import import ITX

yyyy = date.today().year


class GCWerks_Import:

    def __init__(self, options, incoming_dir='chroms_itx'):
        self.options = vars(options)
        self.site = self.options['site']
        self.yyyy = self.options['year']
        self.gcdir = f'/hats/gc/{self.site}'
        self.incoming = Path(f'{self.gcdir}/{str(self.yyyy)[2:4]}/{incoming_dir}')
        # GCWerks program to ingest external chromatogram data
        self.chromatogram_import = Path('/hats/gc/gcwerks-3/bin/chromatogram_import')

    def import_itx(self, itx_file):
        """ Import a single ITX file (all chroms) into GCwerks
            Apply filters and smoothing
        """
        print(itx_file.name)
        itx = ITX(itx_file)

        # apply spike filter before SG smoothing
        if ('s', True) in self.options.items():
            itx.spike_filter('all')

        # apply Savitzky Golay smoothing
        if ('g', True) in self.options.items():
            win = self.options['SGwin']
            ord = self.options['SGorder']
            itx.savitzky_golay('all', winsize=win, order=ord)

        # sends the itx data to the chromatogram_import program
        proc = run([self.chromatogram_import, '-gcdir', self.gcdir],
            input=itx.write(stdout=False), text=True, capture_output=True)

        itx.compress_to_Z(itx_file)

    def recursive_itx_import(self):
        """ Recursive glob finds all itx files in the incoming path.
            This method also uses multiprocessing
        """
        #num_workers = mp.cpu_count()
        num_workers = 10    # faster
        pool = mp.Pool(num_workers)
        for file in self.incoming.rglob('*.itx'):
            pool.apply_async(self.import_itx, args=(file,))
        pool.close()
        pool.join()

    def main(self):
        self.recursive_itx_import()

        # updates integration and mixing ratios
        call(['/hats/gc/gcwerks-3/bin/run-index', '-gcdir', self.gcdir])
        call(['/hats/gc/gcwerks-3/bin/gcupdate', '-gcdir', self.gcdir])
        call(['/hats/gc/gcwerks-3/bin/gccalc', '-gcdir', self.gcdir])


if __name__ == '__main__':

    yyyy = date.today().year
    SGwin, SGorder = 21, 4      # Savitzky Golay default variables
    site = 'agc1'

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format for the FE3 instrument.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default is False)')
    parser.add_argument('-g', action='store_true', default=True,
        help='Apply Savitzky Golay smoothing (default is False)')
    parser.add_argument('-gw', action='store', dest='SGwin', metavar='Win',
        default=SGwin, type=int,
        help=f'Sets Savitzky Golay smoothing window (default = {SGwin} points)')
    parser.add_argument('-go', action='store', dest='SGorder', metavar='Order',
        default=SGorder, type=int,
        help=f'Sets Savitzky Golay order of fit (default = {SGorder})')
    parser.add_argument('-year', action='store', default=yyyy,
        help=f'Which year? (default is {yyyy})')
    parser.add_argument('site', nargs='?', default=site,
        help=f'Valid station code (default is {site})')

    args = parser.parse_args()

    fe3 = GCWerks_Import(args, 'incoming')
    fe3.main()
