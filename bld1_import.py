#! /usr/bin/env python

import argparse
from datetime import date

from gcwerksimport import GCwerks_Import


class BLD1_import(GCwerks_Import):
    """ Class hardcoded for stratcore instrument (BLD1) """

    def __init__(self, args):
        self.site = 'bld1'
        super().__init__(self.site, args)

    @staticmethod
    def itx_portnumbers(files):
        ports = [file.name.split('.')[1] for file in files]
        return list(set(ports))


if __name__ == '__main__':

    yyyy = date.today().year
    SGwin, SGorder = 37, 1      # Savitzky Golay default variables
    WSTART = 25                 # Wide spike filter start time

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format for the BLD1 instrument.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default is False)')
    parser.add_argument('-W', action="store", dest='ws_start', default=WSTART,
        help='Apply wide spike filter (default off)')
    parser.add_argument('-g', action='store_true', default=True,
        help='Apply Savitzky Golay smoothing (default is True)')
    parser.add_argument('-gw', action='store', dest='SGwin', metavar='Win',
        default=SGwin, type=int,
        help=f'Sets Savitzky Golay smoothing window (default = {SGwin} points)')
    parser.add_argument('-go', action='store', dest='SGorder', metavar='Order',
        default=SGorder, type=int,
        help=f'Sets Savitzky Golay order of fit (default = {SGorder})')
    parser.add_argument('-year', action='store', default=yyyy,
        help=f'Which year? (default is {yyyy})')
    parser.add_argument('-reimport', action='store_true', default=False,
        help='Reimport all itx files including .Z archived.')

    args = parser.parse_args()

    inst = BLD1_import(args)
    if args.reimport:
        types = ('*.itx', '*.itx.gz', '*.itx.Z')
        inst.main(import_method=inst.import_recursive_itx, types=types)
    else:
        inst.main(import_method=inst.import_recursive_itx)
