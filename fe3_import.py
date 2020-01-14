#! /home/hats/gdutton/anaconda3/bin/python

""" Added the 'wide_spike_filter' routine.  GSD 150102
    Improved the speed of parse_chroms() by factor of 3.  GSD 150218
    Capture IndexError on wide spike filter.  GSD 150508
    Now uses python3  GSD 170104
    linted GSD 191212
    """

import argparse
from pathlib import Path
from datetime import date
from subprocess import run, call
from multiprocessing import Process
import os
import shutil
import gzip

from itx_import import ITX
from gcwerksimport import GCwerks_import


class FE3_import:

    def __init__(self, options):
        self.options = options
        self.yyyy = date.today().year
        self.path_fe3 = Path(f'/hats/gc/fe3/{str(self.yyyy)[0:2]}/incoming')
        self.gcdir = '/hats/gc/agc1'
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
            win = self.options.get('SGwin')
            ord = self.options.get('SGorder')
            itx.savitzky_golay('all', winsize=win, order=ord)

        proc = run([self.chromatogram_import, '-gcdir', self.gcdir],
            input=itx.write(stdout=False), text=True, capture_output=True)

        itx.compress_to_Z(itx_file)

    def recursive_itx_import(self):
        """ Recursive glob finds all itx files in path_fe3 this
            method also uses multiprocessing
        """
        for file in self.path_fe3.rglob('*.itx'):
            p = Process(target=self.import_itx, args=(file,))
            p.start()

    def main(self):
        self.recursive_itx_import()

        # updates integration and mixing ratios
        call(['/hats/gc/gcwerks-3/bin/run-index', '-gcdir', self.gcdir])
        call(['/hats/gc/gcwerks-3/bin/gcupdate', '-gcdir', self.gcdir])
        call(['/hats/gc/gcwerks-3/bin/gccalc', '-gcdir', self.gcdir])


if __name__ == '__main__':

    SGwin, SGorder = 21, 4      # Savitzky Golay default variables

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default=False)')
    parser.add_argument('-g', action='store_true', default=False,
        help='Apply Savitzky Golay smoothing (default=False)')
    parser.add_argument('-gw', action='store', dest='SGwin', metavar='Win',
        default=SGwin, type=int,
        help='Sets Savitzky Golay smoothing window (default = '+str(SGwin)+' points)')
    parser.add_argument('-go', action='store', dest='SGorder', metavar='Order',
        default=SGorder, type=int,
        help='Sets Savitzky Golay order of fit (default = '+str(SGorder)+')')
    parser.add_argument('-d', action='store', dest='chan', type=int, default=-1,
        help='Display original chrom and exported data for a channel')

    args = parser.parse_args()

    fe3 = FE3_import(vars(args))
    fe3.main()
