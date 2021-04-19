#! /usr/bin/env python

import argparse
from datetime import date
from pathlib import Path
from subprocess import run
import multiprocessing as mp
import os
import shutil
import gzip

import itx_import


class GCwerks_Import:

    def __init__(self, site, options, incoming_dir='chroms_itx'):
        self.options = vars(options)
        self.site = site
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
        itx = itx_import.ITX(itx_file)

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

        itx_import.compress_to_Z(itx_file)

    def import_recursive_itx(self, types=['*.itx', '*.itx.gz']):
        """ Recursive glob finds all itx files in the incoming path.
            This method also uses multiprocessing
        """
        loaded = False
        #num_workers = mp.cpu_count()
        num_workers = 10    # faster
        pool = mp.Pool(num_workers)
        for type in types:
            for file in self.incoming.rglob(type):
                loaded = True
                pool.apply_async(self.import_itx, args=(file,))
        pool.close()
        pool.join()
        return loaded

    def main(self, import_method, *args, **kwargs):
        loaded = import_method(*args, **kwargs)
        if loaded:
            # updates integration and mixing ratios
            run(['/hats/gc/gcwerks-3/bin/run-index', '-gcdir', self.gcdir])
            run(['/hats/gc/gcwerks-3/bin/gcupdate', '-gcdir', self.gcdir])
            run(['/hats/gc/gcwerks-3/bin/gccalc', '-gcdir', self.gcdir])


if __name__ == '__main__':

    SGwin, SGorder = 61, 4      # Savitzky Golay default variables
    WSTART = -1                 # Wide spike filter start time
    yyyy = date.today().year

    opt = argparse.ArgumentParser(
        description="Imports itx files into GCwerks."
    )
    """
    removed options
    opt.add_argument('--all', action="store_true", default=False,
                     help="re-import all itx files in chroms_itx directory.")
    opt.add_argument('-W', action="store", dest='ws_start', default=WSTART,
                     help='Apply wide spike filter (default off)')
    """
    opt.add_argument('-s', action="store_true", default=False,
                     help='Apply 1-point spike filter (default=False)')
    opt.add_argument('-g', action="store_true", default=False,
                     help='Apply Savitzky Golay smoothing (default off)')
    opt.add_argument('-gw', action="store", dest='SGwin', metavar='Win',
                     default=SGwin, type=int,
                     help=f'Sets Savitzky Golay smoothing window (default: {SGwin} points)')
    opt.add_argument('-go', action="store", dest='SGorder', metavar='Order',
                     default=SGorder, type=int,
                     help='Sets Savitzky Golay order of fit (default = '+str(SGorder)+')')
    opt.add_argument("--year", action="store", default=yyyy, dest="year",
                     help=f"the year (default is current year: {yyyy})")
    opt.add_argument("site", help="A station or 'all' for all CATS sites.")

    options = opt.parse_args()

    if options.site.lower() == 'all':
        for s in cats_sites:
            werks = GCWerks_Import(s, options)
            print(f'Working on {s}')
            werks.main()
    else:
        werks = GCwerks_Import(options.site, options)
        werks.main(werks.import_recursive_itx)
